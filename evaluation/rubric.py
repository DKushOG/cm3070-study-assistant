"""The scoring criteria used across the project.

Six criteria, each scored by a human marker from 1 to 5. Keeping them here
means the batch runner and the scoring sheets use the same wording.
"""

CRITERIA = [
    ("accuracy", "Statements are supported by the source material."),
    ("coverage", "Important points across all sources are represented."),
    ("structure", "Notes are organised into the required sections."),
    ("quiz_relevance", "Questions relate to the source material."),
    ("answerability", "Questions can be answered from the notes or sources."),
    ("usefulness", "Output would realistically help a student revise."),
]
