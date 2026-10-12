# Two-entry verification interface

The homepage separates **AI 文案鉴别** (Layer 1 writing patterns) from
**种草推广核验** (promotion and supporting evidence). Existing model contracts
and thresholds are unchanged. Layer numbers are not navigation labels.

Promotion review requires post text. Optional sections accept up to two post
images (2 MB combined), editable comments (up to 30), and a trusted original
nested under each image. Comment screenshots and originals allow 12 MB each.

Screenshot OCR uses `/api/analyze-screenshot`. Candidate comments must be
reviewed before submission. Genuine duplicates are retained. Raw OCR lines
remain accessible for manually correcting missed or ambiguous paragraphs.
Dates terminate comment cards; geometry provides a fallback for undated tails.
This remains a heuristic for username/body/date layouts, not a guarantee for
every screenshot. Clear, cropped comment screenshots work best.

One action submits text, images and reviewed comments to `/api/analyze`, with
`include_report: false`. Trusted-original pairs additionally use
`/api/analyze-layer4-reference`. Findings are displayed independently:
promotional language, comment coordination, image-text alignment (with OCR),
and original-image differences. The interface does not manufacture a combined
promotion probability. Missing evidence and unavailable checks are distinct.

The AI-only entry displays only authorship findings, although the existing
combined backend route still computes its usual response. Switching entries
preserves entered materials. Changes to analyzed content invalidate findings.

## Verification, 2026-10-12

- Modified frontend files passed ESLint; production build and TypeScript passed.
- Three OCR regression tests passed: interface/reply filtering, empty input,
  and long usernames, long bodies, dates, duplicate comments and undated tails.
- A locally rendered Chinese screenshot passed through the real OCR proxy:
  six comments extracted; multiline text merged; date/action text excluded;
  duplicate comments retained; final undated comment preserved.
- Browser upload → editable OCR list → confirmation → combined analysis
  returned six analyzed comments. A reference fixture returned image OCR,
  alignment and original comparison results through their actual endpoints.
- Desktop and mobile layouts checked for horizontal overflow.

These checks validate wiring and these fixtures, not model accuracy on new
social-media content or universal OCR segmentation accuracy.
