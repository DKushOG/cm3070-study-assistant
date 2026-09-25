"""Tests for quiz format validation.

Part of the quiz validation and revalidation group. Two classes near the end
replay real saved runs, so a future change to the parser cannot quietly
reintroduce a fault those runs exposed.
"""
import unittest

from core.generation import QUIZ_PROMPT
from core.quiz_validation import (EMPTY_FAULT, EXPECTED_QUESTION_COUNT,
                                  GENERATED_NOTES_FAULT, INVALID_BASIS_FAULT,
                                  MISSING_FAULT, PLACEHOLDER_FAULT,
                                  PROMPT_PLACEHOLDERS, basis_sources,
                                  has_content, is_placeholder,
                                  looks_like_a_question, names_generated_notes,
                                  parse_quiz, validate_quiz)

# Copied verbatim from the quiz section of
# outputs/20260816_230613_419655_transcript_slide_text_and_notes.txt. This run
# is the source of two of the three defects fixed here: block 3 attributes its
# answer to the generated revision notes, and block 1 is a statement rather
# than a question. It returned four questions, which is a real measurement and
# must keep failing on count.
EVIDENCE_REAL_RUN = """Here are five open short-answer questions based on the source material:

Question: The concept of base rate neglect refers to the tendency to ignore prior probabilities when making decisions.

Suggested answer: This occurs when individuals rely solely on the evidence presented, disregarding the initial probability of the hypothesis before new information is considered.

Question type: Explanation
Source basis: Lecture transcript

Question: What is the main difference between a prior probability and a posterior probability in Bayes theorem?

Suggested answer: A prior probability represents the initial belief or likelihood of a hypothesis before new evidence is considered, whereas a posterior probability represents the updated belief after considering the evidence.

Question type: Explanation
Source basis: Student notes

Question: What is meant by the term "naive" in Naive Bayesian classification?

Suggested answer: The term "naive" refers to the assumption that features are independent of one another, which allows for a simple probabilistic prediction algorithm.

Question type: Definition
Source basis: Revision notes

Question: How does Bayes formula (P(H|E) = P(E|H) * P(H) / P(E)) relate to the concept of likelihood?

Suggested answer: The likelihood of evidence given a hypothesis is a key component of Bayes formula, representing the relationship between the new evidence and the original hypothesis.

Question type: Application
Source basis: Lecture transcript"""

# The output of two real runs, kept word for word so a later change cannot
# quietly bring either failure back. Run 210754 was reported as zero questions
# by the original parser although it held five. Run 211217 echoed the prompt's
# own placeholder wording back as content.
EVIDENCE_UNDER_REPORTED = """Here are five open short-answer questions based on the provided source material:

**Question 1: Definition**
Suggested answer: _____________________________________________________
Source basis: transcript

**Question 2: Explanation**
What is feature engineering in the context of machine learning, and why is it a crucial step? Suggested answer: _____________________________________________________
Source basis: slide text

**Question 3: Comparison**
Compare and contrast Bayes Theorem and Naive Bayes models. How do they differ in their assumptions about features? Suggested answer: _____________________________________________________
Source basis: transcript, slide text, notes

**Question 4: Application**
Describe how you would train a Naive Bayes classifier for sentiment analysis using a multilayer perceptron or another machine learning algorithm as the classifier. Suggested answer: _____________________________________________________
Source basis: notes

**Question 5: Comparison of Term Frequency (TF) and Inverse Document Frequency (IDF)**
Explain how TF and IDF are used together to calculate feature representations in text categorization and information retrieval tasks, highlighting their relative importance. Suggested answer: _____________________________________________________
Source basis: slide text"""

EVIDENCE_PLACEHOLDER_ECHO = """Here are five open short-answer questions based on the provided source material:

Question: the question text
Suggested answer: a concise answer that is supported by the source
Question type: comparison
Source basis: notes

Question: What is the main difference between ignoring prior probabilities in Bayes theorem and using them to update beliefs?

Question: the question text
Suggested answer: a concise answer that is supported by the source
Question type: explanation
Source basis: transcript

Question: When calculating the probability of a hypothesis given the evidence using Bayes theorem, what part of the equation represents the proportion of cells highlighted in the prior probabilities matrix and which part represents the ratio of all the highlighted cells?

Question: the question text
Suggested answer: a concise answer that is supported by the source
Question type: application
Source basis: notes

Question: According to Bayes theorem, what happens when the evidence does not outweigh the prior probability in updating the likelihood of a hypothesis?"""


def make_question(number, question_type="definition"):
    return (f"Question: Question number {number}?\n"
            f"Suggested answer: The answer to {number}.\n"
            f"Question type: {question_type}\n"
            f"Source basis: transcript\n")


def make_quiz(count, question_type="definition"):
    return "\n".join(make_question(n, question_type)
                     for n in range(1, count + 1))


class WellFormedQuizTests(unittest.TestCase):
    """A correctly formatted quiz passes and all four types are accepted."""

    def test_five_valid_questions_pass(self):
        result = validate_quiz(make_quiz(5))
        self.assertTrue(result.passed)
        self.assertEqual(result.questions_found, 5)
        self.assertTrue(result.returned_five)
        self.assertEqual(result.failures, [])

    def test_all_four_question_types_are_accepted(self):
        for question_type in ("definition", "explanation", "comparison",
                              "application"):
            with self.subTest(question_type=question_type):
                self.assertTrue(validate_quiz(make_quiz(5, question_type)).passed)

    def test_expected_count_matches_the_prompt(self):
        self.assertEqual(EXPECTED_QUESTION_COUNT, 5)


class QuestionCountTests(unittest.TestCase):
    """A quiz with the wrong number of questions fails and is counted."""

    def test_four_questions_fail_and_are_counted(self):
        result = validate_quiz(make_quiz(4))
        self.assertFalse(result.passed)
        self.assertEqual(result.questions_found, 4)
        self.assertFalse(result.returned_five)
        self.assertIn("found 4", result.describe())

    def test_six_questions_fail_and_are_counted(self):
        result = validate_quiz(make_quiz(6))
        self.assertFalse(result.passed)
        self.assertEqual(result.questions_found, 6)
        self.assertFalse(result.returned_five)

    def test_empty_output_reports_zero_questions(self):
        result = validate_quiz("")
        self.assertFalse(result.passed)
        self.assertEqual(result.questions_found, 0)
        self.assertFalse(result.returned_five)

    def test_none_output_does_not_crash(self):
        self.assertEqual(validate_quiz(None).questions_found, 0)


class MissingLabelTests(unittest.TestCase):
    """A missing label is reported with the question number it belongs to."""

    def test_missing_source_basis_is_reported_with_its_question_number(self):
        quiz = make_quiz(4) + (
            "Question: Question number 5?\n"
            "Suggested answer: The answer to 5.\n"
            "Question type: definition\n")
        result = validate_quiz(quiz)
        self.assertFalse(result.passed)
        self.assertEqual(result.questions_found, 5)
        self.assertIn("question 5 missing 'Source basis:'", result.failures)

    def test_missing_suggested_answer_is_reported(self):
        quiz = ("Question: Only question?\n"
                "Question type: definition\n"
                "Source basis: transcript\n")
        result = validate_quiz(quiz)
        self.assertIn("Suggested answer", result.describe())


class QuestionTypeTests(unittest.TestCase):
    """A type outside the four categories fails.

    A missing type is reported once, as missing, rather than twice.
    """

    def test_invalid_type_is_reported(self):
        result = validate_quiz(make_quiz(5, "multiple choice"))
        self.assertFalse(result.passed)
        # The count is fine and only the type is wrong.
        self.assertTrue(result.returned_five)
        self.assertIn("invalid type", result.describe())

    def test_type_with_trailing_punctuation_is_accepted(self):
        self.assertTrue(validate_quiz(make_quiz(5, "Definition.")).passed)

    def test_missing_type_is_not_double_reported_as_invalid(self):
        """A missing type line is one fault, reported once as a missing
        label, so failure statistics are not inflated."""
        quiz = ("Question: Only question?\n"
                "Suggested answer: An answer.\n"
                "Source basis: transcript\n")
        result = validate_quiz(quiz)
        type_failures = [f for f in result.failures if "invalid type" in f]
        self.assertEqual(type_failures, [])


class ParsingToleranceTests(unittest.TestCase):
    """List numbering, bold markers and a preamble do not stop the parse."""

    def test_list_numbering_and_bold_markers_are_tolerated(self):
        quiz = ("1. **Question:** What is Bayes theorem?\n"
                "   **Suggested answer:** A rule for updating belief.\n"
                "   **Question type:** definition\n"
                "   **Source basis:** transcript\n")
        blocks = parse_quiz(quiz)
        self.assertEqual(len(blocks), 1)
        self.assertTrue(blocks[0].is_complete)
        self.assertTrue(blocks[0].type_is_valid)

    def test_preamble_before_the_first_question_is_ignored(self):
        quiz = "Here are your questions:\n\n" + make_quiz(5)
        self.assertEqual(validate_quiz(quiz).questions_found, 5)


class NumberedLabelTests(unittest.TestCase):
    """A label with its number inside it still starts a block.

        This is the fault that made a quiz of five questions report zero.

    """

    def test_numbered_and_decorated_labels_start_a_block(self):
        for line in ("**Question 1: Definition**", "Question 2:",
                     "### Question 3:", "3) Question:"):
            with self.subTest(line=line):
                self.assertEqual(len(parse_quiz(line)), 1)

    def test_a_bare_type_word_is_not_credited_as_question_text(self):
        """'**Question 1: Definition**' has a category name where the question
        belongs. It is a heading, so the question text counts as absent and no
        question type is inferred from it."""
        result = validate_quiz("**Question 1: Definition**")
        block = result.blocks[0]
        self.assertIn("Question", block.empty_labels)
        # The type line was never written, so it must be reported missing
        # rather than filled in from the heading.
        self.assertIn("Question type", block.missing_labels)
        self.assertEqual(block.declared_type, "")

    def test_a_real_question_starting_with_a_type_word_is_kept(self):
        """The rejection rule is narrow: only a lone type name is refused."""
        result = validate_quiz(
            "Question 5: Comparison of Term Frequency and Inverse "
            "Document Frequency")
        self.assertNotIn("Question", result.blocks[0].empty_labels)

    def test_prose_mentioning_questions_does_not_create_blocks(self):
        """Ordinary prose must not invent questions, since the count is the
        headline measurement."""
        prose = ("Here are the topics covered:\n"
                 "1. What are some common NLP techniques?\n"
                 "2. How can lexicons contribute to accuracy?\n")
        self.assertEqual(validate_quiz(prose).questions_found, 0)


class MidLineLabelTests(unittest.TestCase):
    """An attribute label is read mid-line, but a question label is not."""

    def test_answer_label_is_recognised_mid_line(self):
        """Real output ran the answer label on after the question text."""
        quiz = ("Question: What is TF-IDF? Suggested answer: A weighting "
                "scheme.\nQuestion type: definition\nSource basis: notes")
        block = parse_quiz(quiz)[0]
        self.assertEqual(block.fields["Question"], "What is TF-IDF?")
        self.assertEqual(block.fields["Suggested answer"],
                         "A weighting scheme.")

    def test_question_label_is_not_recognised_mid_line(self):
        """Anchored deliberately: starting a block from mid-line would let
        prose invent questions."""
        self.assertEqual(
            validate_quiz("Please answer the Question: below").questions_found,
            0)

    def test_unlabelled_line_continues_the_open_value(self):
        quiz = ("Question: First part of the question\n"
                "second part of the question\n"
                "Suggested answer: An answer.\n"
                "Question type: definition\nSource basis: notes")
        block = parse_quiz(quiz)[0]
        self.assertIn("First part", block.fields["Question"])
        self.assertIn("second part", block.fields["Question"])


class EmptyValueTests(unittest.TestCase):
    """A line that is present but says nothing is its own kind of failure.

        Absent, blank and echoed are counted separately, because the evaluation
        reports them separately.

    """

    def test_underscore_answer_is_empty_not_missing(self):
        quiz = ("Question: What is TF-IDF?\n"
                "Suggested answer: ______________________\n"
                "Question type: definition\nSource basis: notes")
        block = validate_quiz(quiz).blocks[0]
        self.assertIn("Suggested answer", block.empty_labels)
        self.assertNotIn("Suggested answer", block.missing_labels)

    def test_punctuation_only_value_has_no_content(self):
        for value in ("", "   ", "____", "---", "...", "??"):
            with self.subTest(value=value):
                self.assertFalse(has_content(value))

    def test_real_content_is_accepted(self):
        for value in ("A weighting scheme.", "definition", "5"):
            with self.subTest(value=value):
                self.assertTrue(has_content(value))

    def test_empty_and_missing_are_reported_with_different_wording(self):
        quiz = ("Question: What is TF-IDF?\n"
                "Suggested answer: ____\n"
                "Question type: definition")
        result = validate_quiz(quiz)
        text = result.describe()
        self.assertIn("has an empty 'Suggested answer:'", text)
        self.assertIn("missing 'Source basis:'", text)

    def test_fault_kinds_are_counted_separately(self):
        quiz = ("Question: What is TF-IDF?\n"
                "Suggested answer: ____\n"
                "Question type: definition")
        result = validate_quiz(quiz)
        self.assertEqual(result.count_faults(EMPTY_FAULT), 1)
        self.assertEqual(result.count_faults(MISSING_FAULT), 1)
        self.assertEqual(result.count_faults(PLACEHOLDER_FAULT), 0)


class PlaceholderEchoTests(unittest.TestCase):
    """Wording copied from the prompt is a fault, not an answer."""

    def test_placeholders_are_derived_from_the_prompt(self):
        """Derived, not copied, so a prompt reword cannot leave this pointing
        at wording that no longer exists."""
        self.assertTrue(PROMPT_PLACEHOLDERS)
        for value in PROMPT_PLACEHOLDERS.values():
            self.assertIn(value, QUIZ_PROMPT)

    def test_expected_placeholder_phrases_are_present(self):
        self.assertEqual(PROMPT_PLACEHOLDERS["Question"], "the question text")
        self.assertEqual(PROMPT_PLACEHOLDERS["Suggested answer"],
                         "a concise answer that is supported by the source")

    def test_echoed_placeholder_is_a_fault(self):
        quiz = ("Question: the question text\n"
                "Suggested answer: a concise answer that is supported by "
                "the source\nQuestion type: comparison\n"
                "Source basis: notes")
        block = validate_quiz(quiz).blocks[0]
        self.assertIn("Question", block.placeholder_labels)
        self.assertIn("Suggested answer", block.placeholder_labels)

    def test_a_real_question_is_not_flagged_as_a_placeholder(self):
        self.assertFalse(
            is_placeholder("Question",
                           "What is the question text used for in a quiz?"))


class EvidenceOneTests(unittest.TestCase):
    """Run 210754, replayed.

        Five readable questions that the older parser reported as zero, which
        hid the faults the run really had.

    """

    def setUp(self):
        self.result = validate_quiz(EVIDENCE_UNDER_REPORTED)

    def test_five_questions_are_found_not_zero(self):
        self.assertEqual(self.result.questions_found, 5)
        self.assertTrue(self.result.returned_five)

    def test_no_question_type_on_any_block(self):
        for block in self.result.blocks:
            self.assertIn("Question type", block.missing_labels)

    def test_every_suggested_answer_is_present_but_empty(self):
        for block in self.result.blocks:
            self.assertIn("Suggested answer", block.empty_labels)
            self.assertNotIn("Suggested answer", block.missing_labels)

    def test_question_one_has_no_question_text(self):
        self.assertIn("Question", self.result.blocks[0].empty_labels)

    def test_later_questions_keep_their_question_text(self):
        for block in self.result.blocks[1:]:
            self.assertNotIn("Question", block.empty_labels)
        self.assertIn("feature engineering",
                      self.result.blocks[1].fields["Question"])

    def test_the_run_still_fails_overall(self):
        self.assertFalse(self.result.passed)


class EvidenceTwoTests(unittest.TestCase):
    """Run 211217, replayed.

        The model echoed the prompt's placeholder wording, which the older
        validator counted as real content.

    """

    def setUp(self):
        self.result = validate_quiz(EVIDENCE_PLACEHOLDER_ECHO)

    def test_six_questions_are_found(self):
        self.assertEqual(self.result.questions_found, 6)
        self.assertFalse(self.result.returned_five)

    def test_placeholder_echo_reported_on_the_three_blocks(self):
        echoing = [block.number for block in self.result.blocks
                   if block.placeholder_labels]
        self.assertEqual(echoing, [1, 3, 5])

    def test_real_questions_are_not_flagged_as_placeholders(self):
        for number in (2, 4, 6):
            block = self.result.blocks[number - 1]
            self.assertEqual(block.placeholder_labels, [])

    def test_the_run_still_fails_overall(self):
        self.assertFalse(self.result.passed)


class SourceBasisTests(unittest.TestCase):
    """The source basis must name an input the student supplied.

        Naming the system's own revision notes is its own fault, since that is
        a different failure from naming nothing.

    """

    def test_real_wordings_resolve_to_their_input_source(self):
        cases = {
            "transcript": ["transcript"],
            "Lecture transcript": ["transcript"],
            "notes": ["notes"],
            "student notes": ["notes"],
            "Student notes": ["notes"],
            "slide text": ["slide text"],
            "Slide text": ["slide text"],
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(basis_sources(value), expected)

    def test_combinations_resolve_to_every_source_named(self):
        self.assertEqual(basis_sources("notes, slide text"),
                         ["notes", "slide text"])
        self.assertEqual(basis_sources("transcript, slide text, notes"),
                         ["notes", "slide text", "transcript"])

    def test_generated_notes_is_recognised_and_not_read_as_student_notes(self):
        self.assertTrue(names_generated_notes("Revision notes"))
        # The trailing "notes" must not be counted as the student notes input.
        self.assertEqual(basis_sources("Revision notes"), [])

    def test_wording_naming_no_source_resolves_to_nothing(self):
        for value in ("general knowledge", "the internet", "my own reasoning"):
            with self.subTest(value=value):
                self.assertEqual(basis_sources(value), [])

    def test_valid_single_source_is_not_a_fault(self):
        quiz = ("Question: What is TF-IDF?\nSuggested answer: A weighting.\n"
                "Question type: definition\nSource basis: Lecture transcript")
        self.assertEqual(validate_quiz(quiz).blocks[0].basis_faults, [])

    def test_valid_combination_is_not_a_fault(self):
        quiz = ("Question: What is TF-IDF?\nSuggested answer: A weighting.\n"
                "Question type: definition\nSource basis: notes, slide text")
        self.assertEqual(validate_quiz(quiz).blocks[0].basis_faults, [])

    def test_generated_notes_basis_is_a_distinct_named_fault(self):
        quiz = ("Question: What is TF-IDF?\nSuggested answer: A weighting.\n"
                "Question type: definition\nSource basis: Revision notes")
        result = validate_quiz(quiz)
        self.assertIn(GENERATED_NOTES_FAULT, result.blocks[0].basis_faults)
        self.assertNotIn(INVALID_BASIS_FAULT, result.blocks[0].basis_faults)
        self.assertEqual(result.count_faults(GENERATED_NOTES_FAULT), 1)
        self.assertIn("generated revision notes", result.describe())

    def test_basis_naming_no_source_is_the_other_fault(self):
        quiz = ("Question: What is TF-IDF?\nSuggested answer: A weighting.\n"
                "Question type: definition\nSource basis: general knowledge")
        result = validate_quiz(quiz)
        self.assertIn(INVALID_BASIS_FAULT, result.blocks[0].basis_faults)
        self.assertEqual(result.count_faults(INVALID_BASIS_FAULT), 1)

    def test_absent_basis_is_not_double_reported(self):
        """A missing label is already a fault, so it must not also count as
        an invalid basis, or the figures would be inflated."""
        quiz = ("Question: What is TF-IDF?\nSuggested answer: A weighting.\n"
                "Question type: definition")
        result = validate_quiz(quiz)
        self.assertEqual(result.blocks[0].basis_faults, [])
        self.assertEqual(result.count_faults(INVALID_BASIS_FAULT), 0)

    def test_basis_fault_makes_validation_fail(self):
        quiz = "\n\n".join(
            f"Question: Q{n}?\nSuggested answer: A{n}.\n"
            f"Question type: definition\nSource basis: Revision notes"
            for n in range(1, 6))
        self.assertFalse(validate_quiz(quiz).passed)


class QuestionShapeTests(unittest.TestCase):
    """The question-shape check is a warning, not a fault.

        It is counted and reported but left out of the verdict, so the pass
        rate earlier results were collected under does not move.

    """

    def test_question_mark_passes(self):
        self.assertTrue(looks_like_a_question("What is Bayes theorem?"))

    def test_interrogative_opener_without_a_mark_passes(self):
        for text in ("Explain how Bayes theorem works",
                     "Compare prior and posterior probability",
                     "In what way does evidence update belief"):
            with self.subTest(text=text):
                self.assertTrue(looks_like_a_question(text))

    def test_declarative_sentence_is_flagged(self):
        statement = ("The concept of base rate neglect refers to the tendency "
                     "to ignore prior probabilities when making decisions.")
        self.assertFalse(looks_like_a_question(statement))

    def test_warning_is_reported_separately_from_failures(self):
        quiz = "\n\n".join(
            f"Question: Statement number {n} is a fact.\n"
            f"Suggested answer: A{n}.\n"
            f"Question type: definition\nSource basis: transcript"
            for n in range(1, 6))
        result = validate_quiz(quiz)
        self.assertEqual(result.shape_warning_count, 5)
        # Five well-formed blocks with a valid basis: the shape warning must
        # not turn this into a failure.
        self.assertTrue(result.passed)
        self.assertEqual(result.failures, [])
        self.assertIn("does not read as a question",
                      result.describe_warnings())

    def test_verdict_is_identical_with_and_without_the_warning(self):
        questioning = "\n\n".join(
            f"Question: What is concept {n}?\nSuggested answer: A{n}.\n"
            f"Question type: definition\nSource basis: transcript"
            for n in range(1, 6))
        declarative = "\n\n".join(
            f"Question: Concept {n} is a fact.\nSuggested answer: A{n}.\n"
            f"Question type: definition\nSource basis: transcript"
            for n in range(1, 6))
        with_warning = validate_quiz(declarative)
        without_warning = validate_quiz(questioning)
        self.assertEqual(with_warning.passed, without_warning.passed)
        self.assertEqual(with_warning.shape_warning_count, 5)
        self.assertEqual(without_warning.shape_warning_count, 0)


class RealRunTests(unittest.TestCase):
    """A real run kept word for word.

        Replayed so a later change to the parser cannot quietly undo any of the
        fixes above.

    """

    def setUp(self):
        self.result = validate_quiz(EVIDENCE_REAL_RUN)

    def test_still_reports_four_questions(self):
        """The model returning four is a real measurement, not something the
        parser should paper over."""
        self.assertEqual(self.result.questions_found, 4)
        self.assertFalse(self.result.returned_five)
        self.assertIn("expected 5 questions, found 4", self.result.failures)

    def test_revision_notes_basis_is_flagged_on_block_three(self):
        flagged = [block.number for block in self.result.blocks
                   if GENERATED_NOTES_FAULT in block.basis_faults]
        self.assertEqual(flagged, [3])

    def test_the_other_blocks_have_a_valid_basis(self):
        for number in (1, 2, 4):
            block = self.result.blocks[number - 1]
            self.assertEqual(block.basis_faults, [])
            self.assertTrue(block.basis_is_valid)

    def test_declarative_first_question_is_warned_about(self):
        warned = [block.number for block in self.result.blocks
                  if block.shape_warnings]
        self.assertEqual(warned, [1])

    def test_all_four_labels_are_present_on_every_block(self):
        """The run was well formed apart from the defects under test."""
        for block in self.result.blocks:
            self.assertEqual(block.missing_labels, [])
            self.assertEqual(block.empty_labels, [])
            self.assertEqual(block.placeholder_labels, [])

    def test_shape_warning_does_not_change_the_verdict(self):
        self.assertEqual(self.result.shape_warning_count, 1)
        self.assertFalse(self.result.passed)
        # It fails on count and basis, never on the warning.
        self.assertNotIn("does not read as a question",
                         "; ".join(self.result.failures))


if __name__ == "__main__":
    unittest.main()
