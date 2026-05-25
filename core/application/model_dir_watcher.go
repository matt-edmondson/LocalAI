package application

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/mudler/LocalAI/core/config"
	"github.com/mudler/xlog"
)

// modelDirWatcher watches the models directory for new or modified YAML
// configuration files and loads them into the ModelConfigLoader without
// restarting LocalAI.
type modelDirWatcher struct {
	appConfig *config.ApplicationConfig
	cl        *config.ModelConfigLoader
	mu        sync.Mutex
	known     map[string]time.Time // abs file path → last mod time
}

func newModelDirWatcher(appConfig *config.ApplicationConfig, cl *config.ModelConfigLoader) *modelDirWatcher {
	return &modelDirWatcher{
		appConfig: appConfig,
		cl:        cl,
		known:     make(map[string]time.Time),
	}
}

// scan reads the models directory and loads any new or modified YAML files.
func (w *modelDirWatcher) scan() {
	modelsPath := w.appConfig.SystemState.Model.ModelsPath
	entries, err := os.ReadDir(modelsPath)
	if err != nil {
		xlog.Warn("model dir watcher: cannot read models path", "path", modelsPath, "error", err)
		return
	}

	opts := w.appConfig.ToConfigLoaderOptions()
	w.mu.Lock()
	defer w.mu.Unlock()

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		ext := strings.ToLower(filepath.Ext(name))
		if (ext != ".yaml" && ext != ".yml") || strings.HasPrefix(name, ".") {
			continue
		}

		filePath := filepath.Join(modelsPath, name)
		info, err := entry.Info()
		if err != nil {
			continue
		}
		modTime := info.ModTime()

		lastSeen, alreadyKnown := w.known[filePath]
		if alreadyKnown && !modTime.After(lastSeen) {
			continue // unchanged
		}

		if err := w.cl.ReadModelConfig(filePath, opts...); err != nil {
			xlog.Debug("model dir watcher: skipping file", "file", filePath, "error", err)
			continue
		}

		if !alreadyKnown {
			xlog.Info("model dir watcher: loaded new model config", "file", filePath)
		} else {
			xlog.Info("model dir watcher: reloaded modified model config", "file", filePath)
		}
		w.known[filePath] = modTime
	}
}

// startModelDirWatcher starts background watching of the models directory so
// that YAML configs written by ModelMan (or any external tool) are picked up
// without restarting LocalAI.
//
// Polling is started when ApplicationConfig.DynamicConfigsDirPollInterval > 0
// (required for NFS mounts where inotify is unreliable). fsnotify is also
// attempted as a best-effort immediate notification for local filesystems.
func startModelDirWatcher(appConfig *config.ApplicationConfig, cl *config.ModelConfigLoader) {
	modelsPath := appConfig.SystemState.Model.ModelsPath
	if modelsPath == "" {
		return
	}

	watcher := newModelDirWatcher(appConfig, cl)

	// Polling fallback — required for NFS / inotify-unfriendly mounts.
	if appConfig.DynamicConfigsDirPollInterval > 0 {
		xlog.Info("model dir watcher: polling enabled", "path", modelsPath, "interval", appConfig.DynamicConfigsDirPollInterval)
		ticker := time.NewTicker(appConfig.DynamicConfigsDirPollInterval)
		go func() {
			for range ticker.C {
				watcher.scan()
			}
		}()
	}

	// fsnotify for local filesystems (best-effort; silently skipped on NFS).
	fsWatcher, err := fsnotify.NewWatcher()
	if err != nil {
		xlog.Debug("model dir watcher: fsnotify unavailable", "error", err)
		return
	}
	if err := fsWatcher.Add(modelsPath); err != nil {
		xlog.Debug("model dir watcher: cannot watch models path with fsnotify", "path", modelsPath, "error", err)
		fsWatcher.Close() //nolint:errcheck
		return
	}

	go func() {
		defer fsWatcher.Close() //nolint:errcheck
		for {
			select {
			case event, ok := <-fsWatcher.Events:
				if !ok {
					return
				}
				if !event.Has(fsnotify.Write | fsnotify.Create) {
					continue
				}
				ext := strings.ToLower(filepath.Ext(event.Name))
				if ext != ".yaml" && ext != ".yml" {
					continue
				}
				if strings.HasPrefix(filepath.Base(event.Name), ".") {
					continue
				}
				watcher.scan()
			case err, ok := <-fsWatcher.Errors:
				if !ok {
					return
				}
				xlog.Debug("model dir watcher: fsnotify error", "error", err)
			}
		}
	}()
}
