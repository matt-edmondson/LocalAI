package worker

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/mudler/LocalAI/core/services/messaging"
)

var _ = Describe("handleBackendStop", func() {
	// The handler runs its body in a goroutine (so a slow stop can't
	// head-of-line-block the subscription); the reply lands asynchronously.
	expectAck := func(payload []byte) {
		s := &backendSupervisor{processes: map[string]*backendProcess{}}
		replies := make(chan []byte, 1)
		s.handleBackendStop(payload, func(data []byte) { replies <- data })

		var raw []byte
		Eventually(replies, "2s").Should(Receive(&raw))
		var reply messaging.BackendStopReply
		Expect(json.Unmarshal(raw, &reply)).To(Succeed())
		Expect(reply.Success).To(BeTrue())
	}

	It("acks after stopping a specific process key", func() {
		// No process under this key — stopBackend no-ops, but the ack must
		// still arrive: the frontend's abandoned-load cleanup blocks on it.
		expectAck([]byte(`{"backend":"my-model#0"}`))
	})

	It("acks the stop-all form (empty payload)", func() {
		expectAck(nil)
	})
})
