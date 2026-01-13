# module mcp_server

# system
import json
from datetime import datetime, timedelta

# libs
from fastmcp import FastMCP

# init mcp server instance
mcp = FastMCP("MongoDB MCP Server")


@mcp.tool()
def nest_info() -> dict:
    """Return basic information about the local NEST installation (version, a few available models)."""
    try:
        import nest

        info = {"nest_version": getattr(nest, "__version__", "unknown")}

        # Try to list some models (API differs across NEST versions)
        models = []
        try:
            models = nest.Models()
        except Exception:
            try:
                models = nest.GetDefaults()
            except Exception:
                models = []

        info["models_sample"] = list(models)[:50] if models else []
        return info
    except Exception as e:
        return {"error": f"NEST not available or failed to import: {e}"}


@mcp.tool()
def run_iaf_single_neuron(
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    poisson_rate_hz: float = 8000.0,
    syn_weight: float = 20.0,
    syn_delay_ms: float = 1.5,
    neuron_model: str = "iaf_psc_alpha",
    record_vm: bool = True,
    vm_downsample: int = 10,
    seed: int = 42,
) -> dict:
    """Run a simple single-neuron NEST simulation (Poisson input -> LIF neuron) and return spikes + basic Vm summary."""
    try:
        import nest
        import numpy as np

        # Reset & configure kernel
        nest.ResetKernel()
        try:
            nest.SetKernelStatus({"resolution": float(dt_ms)})
        except Exception:
            pass

        # Seed (best-effort across versions)
        try:
            nest.SetKernelStatus({"rng_seed": int(seed), "grng_seed": int(seed) + 1})
        except Exception:
            try:
                nest.SetKernelStatus({"seed": int(seed)})
            except Exception:
                pass

        # Create nodes
        neuron = nest.Create(neuron_model, 1)
        pg = nest.Create("poisson_generator", 1, {"rate": float(poisson_rate_hz)})
        sr = nest.Create("spike_recorder", 1)

        mm = None
        if record_vm:
            mm = nest.Create(
                "multimeter",
                1,
                {
                    "record_from": ["V_m"],
                    "interval": float(dt_ms),
                },
            )

        # Connect
        nest.Connect(pg, neuron, syn_spec={"weight": float(syn_weight), "delay": float(syn_delay_ms)})
        nest.Connect(neuron, sr)
        if mm is not None:
            nest.Connect(mm, neuron)

        # Simulate
        nest.Simulate(float(duration_ms))

        # Extract spikes (handle different NEST APIs)
        def _events(node):
            try:
                st = nest.GetStatus(node)[0]
                return st.get("events", {})
            except Exception:
                try:
                    return node.events
                except Exception:
                    return {}

        spike_events = _events(sr)
        spike_times = spike_events.get("times", [])
        spike_senders = spike_events.get("senders", [])

        # Convert to python lists
        spike_times = list(np.array(spike_times).astype(float)) if len(spike_times) else []
        spike_senders = list(np.array(spike_senders).astype(int)) if len(spike_senders) else []

        # Vm
        vm_summary = {}
        vm_sample = []
        if mm is not None:
            mm_events = _events(mm)
            t = mm_events.get("times", [])
            vm = mm_events.get("V_m", [])
            t = np.array(t, dtype=float) if len(t) else np.array([], dtype=float)
            vm = np.array(vm, dtype=float) if len(vm) else np.array([], dtype=float)

            if t.size and vm.size:
                vm_summary = {
                    "vm_min": float(np.min(vm)),
                    "vm_max": float(np.max(vm)),
                    "vm_mean": float(np.mean(vm)),
                    "vm_std": float(np.std(vm)),
                }
                # downsample for payload size
                ds = max(1, int(vm_downsample))
                vm_sample = [{"t_ms": float(t[i]), "V_m": float(vm[i])} for i in range(0, min(len(t), len(vm)), ds)][:2000]

        spike_count = len(spike_times)
        firing_rate_hz = float(spike_count) / (float(duration_ms) / 1000.0) if duration_ms > 0 else 0.0

        return {
            "simulation": "single_neuron_poisson",
            "params": {
                "duration_ms": float(duration_ms),
                "dt_ms": float(dt_ms),
                "poisson_rate_hz": float(poisson_rate_hz),
                "syn_weight": float(syn_weight),
                "syn_delay_ms": float(syn_delay_ms),
                "neuron_model": str(neuron_model),
                "seed": int(seed),
            },
            "results": {
                "spike_count": spike_count,
                "firing_rate_hz": firing_rate_hz,
                "spikes_preview": {
                    "times_ms_first_200": spike_times[:200],
                    "senders_first_200": spike_senders[:200],
                },
                "vm_summary": vm_summary,
                "vm_sample": vm_sample,
            },
        }
    except Exception as e:
        return {"error": f"Simulation failed: {e}"}


@mcp.tool()
def run_ei_microcircuit(
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    n_exc: int = 80,
    n_inh: int = 20,
    p_connect: float = 0.1,
    w_exc: float = 1.0,
    w_inh: float = -5.0,
    delay_ms: float = 1.5,
    ext_rate_hz: float = 8000.0,
    ext_weight: float = 1.0,
    seed: int = 7,
) -> dict:
    """Run a tiny E/I random network with external Poisson drive and return population spike stats."""
    try:
        import nest
        import numpy as np

        nest.ResetKernel()
        try:
            nest.SetKernelStatus({"resolution": float(dt_ms)})
        except Exception:
            pass

        try:
            nest.SetKernelStatus({"rng_seed": int(seed), "grng_seed": int(seed) + 1})
        except Exception:
            try:
                nest.SetKernelStatus({"seed": int(seed)})
            except Exception:
                pass

        # Nodes
        exc = nest.Create("iaf_psc_alpha", int(n_exc))
        inh = nest.Create("iaf_psc_alpha", int(n_inh))

        pg_exc = nest.Create("poisson_generator", 1, {"rate": float(ext_rate_hz)})
        pg_inh = nest.Create("poisson_generator", 1, {"rate": float(ext_rate_hz)})

        sr_exc = nest.Create("spike_recorder", 1)
        sr_inh = nest.Create("spike_recorder", 1)

        # External drive
        nest.Connect(pg_exc, exc, syn_spec={"weight": float(ext_weight), "delay": float(delay_ms)})
        nest.Connect(pg_inh, inh, syn_spec={"weight": float(ext_weight), "delay": float(delay_ms)})

        # Recurrent random connections
        conn = {"rule": "pairwise_bernoulli", "p": float(p_connect)}
        nest.Connect(exc, exc, conn_spec=conn, syn_spec={"weight": float(w_exc), "delay": float(delay_ms)})
        nest.Connect(exc, inh, conn_spec=conn, syn_spec={"weight": float(w_exc), "delay": float(delay_ms)})
        nest.Connect(inh, exc, conn_spec=conn, syn_spec={"weight": float(w_inh), "delay": float(delay_ms)})
        nest.Connect(inh, inh, conn_spec=conn, syn_spec={"weight": float(w_inh), "delay": float(delay_ms)})

        # Record
        nest.Connect(exc, sr_exc)
        nest.Connect(inh, sr_inh)

        nest.Simulate(float(duration_ms))

        def _events(node):
            try:
                st = nest.GetStatus(node)[0]
                return st.get("events", {})
            except Exception:
                try:
                    return node.events
                except Exception:
                    return {}

        ev_exc = _events(sr_exc)
        ev_inh = _events(sr_inh)

        t_exc = np.array(ev_exc.get("times", []), dtype=float)
        t_inh = np.array(ev_inh.get("times", []), dtype=float)

        spikes_exc = int(t_exc.size)
        spikes_inh = int(t_inh.size)

        dur_s = float(duration_ms) / 1000.0 if duration_ms > 0 else 1.0
        rate_exc_hz = float(spikes_exc) / (int(n_exc) * dur_s) if n_exc > 0 else 0.0
        rate_inh_hz = float(spikes_inh) / (int(n_inh) * dur_s) if n_inh > 0 else 0.0

        return {
            "simulation": "ei_microcircuit",
            "params": {
                "duration_ms": float(duration_ms),
                "dt_ms": float(dt_ms),
                "n_exc": int(n_exc),
                "n_inh": int(n_inh),
                "p_connect": float(p_connect),
                "w_exc": float(w_exc),
                "w_inh": float(w_inh),
                "delay_ms": float(delay_ms),
                "ext_rate_hz": float(ext_rate_hz),
                "ext_weight": float(ext_weight),
                "seed": int(seed),
            },
            "results": {
                "spikes_total": int(spikes_exc + spikes_inh),
                "exc": {
                    "spike_count": spikes_exc,
                    "mean_rate_hz_per_neuron": rate_exc_hz,
                    "spike_times_first_200": t_exc[:200].tolist() if t_exc.size else [],
                },
                "inh": {
                    "spike_count": spikes_inh,
                    "mean_rate_hz_per_neuron": rate_inh_hz,
                    "spike_times_first_200": t_inh[:200].tolist() if t_inh.size else [],
                },
            },
        }
    except Exception as e:
        return {"error": f"Simulation failed: {e}"}

if __name__ == '__main__':
    mcp.run(transport='sse', port=8050)
