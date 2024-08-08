# nessai-gw

Gravitational-wave specific proposals and reparameterisations for nessai

## Usage

Once installed, these proposals can be used in `nessai` by specifying the
`flow_proposal_class` keyword argument when using the standard nested sampler.

### Example

```python
fs = FlowSampler(
    model,
    ...,
    flow_proposal_class="gwflowproposal",
)
```
