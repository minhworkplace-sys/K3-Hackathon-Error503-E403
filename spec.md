# Feature Specification: Lecture Slide Quick Summary

## Overview

Add a feature that generates a short summary (maximum 3 sentences) for each lecture slide. The summary helps students quickly understand the content of previous slides so they can catch up with the lecturer without interrupting the class.

---

## Problem Statement

During a lecture, students may miss one or more slides because they are taking notes, distracted, or temporarily lose focus. Reading previous slides while the lecturer continues speaking causes them to fall even further behind.

Students need a fast way to understand what they missed without reviewing the entire slide.

---

## Goal

Provide a concise AI-generated summary for each lecture slide that can be read within a few seconds.

---

# User Stories

### Student

As a student,
I want to view a short summary of any previous slide,
so that I can quickly catch up with the lecture without reading the entire slide deck.

### Lecturer

As a lecturer,
I want students to quickly understand missed content,
so they can stay engaged with the ongoing lecture.

---

# Functional Requirements

## FR1. Generate Slide Summary

The system shall generate a concise summary for every lecture slide.

### Requirements

- Maximum 3 sentences.
- Focus only on the key ideas.
- Ignore decorative or irrelevant content.
- Use simple and easy-to-understand language.

---

## FR2. Display Summary

Each slide shall display a "Summary" section.

The summary should:

- Be clearly separated from the slide content.
- Be readable on both desktop and mobile.
- Load automatically after slide processing is completed.

---

## FR3. Summary Regeneration

If the slide content changes,

the system shall regenerate the summary automatically.

---

## FR4. AI Failure Handling

If summary generation fails,

the system shall

- display an error message,
- allow the user to retry,
- never display incomplete or corrupted summaries.

---

# Non-functional Requirements

### Performance

- Summary generation should complete within 5 seconds after slide upload.

### Readability

- Summary length ≤ 3 sentences.
- Language should match the lecture language.

### Reliability

- The system should successfully generate summaries for at least 95% of uploaded slides.

---

# Acceptance Criteria

### AC1

Given a lecture slide,

when processing is completed,

then a summary is displayed.

---

### AC2

The summary contains no more than three sentences.

---

### AC3

The summary accurately captures the main ideas of the slide.

---

### AC4

If the slide is updated,

a new summary is generated automatically.

---

### AC5

If AI processing fails,

the user sees an appropriate error message and can retry.

---

# Out of Scope

This feature does NOT include:

- Summarizing multiple slides together.
- Audio transcription.
- Lecture recording summary.
- Question answering.
- Translation.

---

# Success Metrics

- 95% of slides successfully generate summaries.
- Average generation time < 5 seconds.
- At least 80% of surveyed students report that the summary helps them catch up with the lecture.
- Average summary length ≤ 3 sentences.