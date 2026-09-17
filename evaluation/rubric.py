"""Evaluation rubric used across the project.

The six criteria are the ones defined in the Design chapter and applied in
the Preliminary Project Report evaluation. Scores are given by a human
marker on a 1 to 5 scale. Keeping the definitions in code means the batch
runner and the report always agree on the criteria.
"""

CRITERIA = [
    ("accuracy", "Statements are supported by the source material."),
    ("coverage", "Important points across all sources are represented."),
    ("structure", "Notes are organised into the required sections."),
    ("quiz_relevance", "Questions relate to the source material."),
    ("answerability", "Questions can be answered from the notes or sources."),
    ("usefulness", "Output would realistically help a student revise."),
]
