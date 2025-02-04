nessai-gw
=========

:code:`nessai-gw` provides gravitational-wave specific functionality for the
`nessai` nested sampling package.


Quickstart
-----------

Once installed, the proposal classes provided by :code``nessai-gw`` can be used
in the same way as the default proposal classes in :code:`nessai`:


.. code-block:: python

   from nessai.flowsampler import FlowSampler

   model = ...

   sampler = Sampler(
      model,
      flow_proposal_class='GWFlowProposal',
      ...
    )


.. toctree::
   :hidden:
   :maxdepth: 2

   index
   installation
   proposals
