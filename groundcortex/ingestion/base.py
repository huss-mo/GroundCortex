from __future__ import annotations

from abc import ABC, abstractmethod

from groundcortex.pipeline.models import Experience


class IngestionAdapter(ABC):
    @abstractmethod
    def ingest(self) -> list[Experience]:
        """Read source(s), apply file-level hash check, return new pending Experiences.

        Returns an empty list if nothing changed since the last ingest.
        Superseding stale experiences and upserting source_files records are
        handled inside each adapter implementation via the shared parse helper.
        """
