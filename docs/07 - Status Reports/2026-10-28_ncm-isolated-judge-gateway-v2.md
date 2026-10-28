# Isolated gateway launcher correction

The v1 launch exited before starting its gateway: the service PATH did not
contain the nested bare `pixi` executable. The recorded invocation ended with
exit1 and no model calls. Preserve its code, producer and failure observation.

Use a fresh v2 service and probe root, the same loopback8092 port and complete
v1 neutral availability protocol, with an absolute `/home/rabbit/.pixi/bin/pixi`
launcher. The gateway retains its existing locked Pixi environment, exact model
names, source bindings, nonpersistent completion processes and single worker.
The supervised service has a2GiB memory ceiling and no automatic restart.
Run the same three process-identity, response-model and immutable-source tests
against the v2 implementation before launch. No semantic qualification or
application grading is authorized by these neutral probes alone.
