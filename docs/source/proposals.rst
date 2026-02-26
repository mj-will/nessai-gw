Proposals
=========

The available proposal classes are:

- :code:`GWFlowProposal`: A proposal class that wraps the :code:`FlowProposal` and includes default reparameterisations.
- :code:`AugmentedGWFlowProposal`: A proposal class that wraps the :code:`AugmentFlowproposal` and includes the same reparameterisations as :code:`GWFlowProposal`.


Experimental Proposals
----------------------

.. warning::

    The following proposal classes are experimental and may change or not be supported in future versions of :code:`nessai-gw`.

It also includes the following experimental proposal classes:

- :code:`ClusteringGWFlowProposal`: A proposal class that wraps the experimental :code:`ClusteringFlowProposal` and includes the same reparameterisations as :code:`GWFlowProposal`.
- :code:`MCMCGWFlowProposal`: A proposal class that wraps the experimental :code:`MCMCFlowProposal` and includes the same reparameterisations as :code:`GWFlowProposal`.
