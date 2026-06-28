CaVE Datasets
+++++++++++++

CaVE trains on the binding constraints at the true optimum. These labels are prepared by ``optDatasetConstrs`` and consumed by the ``CaVE`` loss.

Use these pages together:

* :doc:`../getting_started/data` explains ``optDatasetConstrs`` and the custom collate function for ragged binding-constraint matrices.
* :doc:`../getting_started/function` explains the ``CaVE`` loss, its projection step, and its solver requirements.
* :doc:`../notebooks` links to the CaVE Colab walkthrough.

CaVE currently targets binary linear programs and requires a Gurobi-backed ``optModel`` when extracting binding-constraint normals.
