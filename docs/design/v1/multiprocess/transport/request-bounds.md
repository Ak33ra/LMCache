# Opt-in CPU request bounds

`MPServerConfig.request_limits` supplies immutable `RequestLimits` to the ZMQ
server. `None` preserves existing transport behavior. Partition activation
selects finite payload, accepted-request, and tensor-workspace limits.
These auxiliary limits are separate from managed cache backing and SDK scratch.

One semaphore credit covers a decoded request waiting in an executor, a running
handler, and its serialized response waiting in the Python output queue. The
socket loop releases the credit only after nonblocking send or drop. Handler
failure and pre-submit decoding failure return refusal and release their credit.
The response queue has the same finite capacity as the semaphore. A completed
handler cannot fill it beyond that capacity because it still owns its credit.

Oversized aggregate wire requests are refused before decoding. The limit counts
routing and request-ID frames as well as payloads. Serialized responses larger
than the limit are replaced by refusal. Unary requests only are supported in
bounded mode. Stream subscriptions and stream control frames are refused.
Refusal uses the reserved `__request_refused__` response operation, which existing
clients report as an operation mismatch. Refusals are sent directly with
`NOBLOCK`, never appended to another queue. Backpressure/disconnection can drop
a response or refusal; callers need finite timeouts. A timeout does not cancel
a running handler or free its credit early.

The single socket loop still owns sends and receives. Handler callbacks only
append to the queue and notify the loop. Bounded shutdown stops admission, waits
for accepted handlers, then discards remaining output and closes the notifier.
A blocked handler keeps shutdown pending. Repeated sequential close is harmless.

## Scope of the memory bound

For limit N accepted requests and P wire bytes, retained Python wire responses
are bounded by N times P, plus bounded routing metadata. Accepted requests are
at most N, each with at most P encoded bytes before decoding. Serialization can
briefly allocate the encoded result before its size is checked. Typed-inline
handlers must enforce their separate W workspace limit before decode/copy/clone,
including the sum of concurrently live tensor copies, then bound returned bytes.
`RequestLimits.check_workspace_bytes` supplies this validation; the transport
cannot infer tensor allocation sizes from arbitrary handler return objects.

ZMQ send/receive HWM is N messages per peer and MAXMSGSIZE is P per frame.
These are not global byte/RSS bounds. The aggregate request check happens after
`recv_multipart`, so a rejected multipart request can transiently exceed P.
Decoded metadata, libzmq buffers, kernel buffers, connection count, worker/client
queues, runtime overhead, and arbitrary extension handlers are outside the
accepted-request accounting. This path assumes controlled typed-inline clients;
it is not protection against malicious multipart or connection floods.

## Verification

`tests/v1/multiprocess/test_request_limits.py` uses real loopback TCP ROUTER and
DEALER sockets with 1 KiB wire limits. It verifies accepted running/queued work,
aggregate oversized requests, malformed arguments, handler failures, oversized
responses, reuse after refusal, and shutdown while a handler is blocked.
`USER_VERIFY.md` records the exact command and dated evidence.
