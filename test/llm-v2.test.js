'use strict';
/**
 * llm-v2.test.js — Tests for new llmClient.js v2 APIs (pure, no sidecar).
 *
 * Tests parseStructured(), buildMatchPrompt(), buildAssessPrompt().
 * These are all pure functions — no I/O, no init() required.
 */

const { ok, equal } = global._assert;

const llm = require('../src/llmClient');

module.exports = function () {

  // ── parseStructured ───────────────────────────────────────────────────────

  const sampleText = `STRENGTHS:
- Strong React experience with TypeScript
- AWS certification (Solutions Architect)
- Kubernetes deployment experience

GAPS:
- No experience with GraphQL
- Limited Python skills

RECOMMENDATION:
Shortlist this candidate with a caveat around GraphQL. Pair with a senior engineer for the GraphQL onboarding.`;

  const parsed = llm.parseStructured(sampleText, ['STRENGTHS', 'GAPS', 'RECOMMENDATION']);

  ok(typeof parsed === 'object' && parsed !== null,        'parseStructured: returns object');
  ok(Array.isArray(parsed['STRENGTHS']),                   'parseStructured: STRENGTHS is array');
  ok(Array.isArray(parsed['GAPS']),                        'parseStructured: GAPS is array');

  ok(parsed['STRENGTHS'].length >= 2,                      'parseStructured: extracted STRENGTHS items');
  ok(parsed['GAPS'].length >= 1,                           'parseStructured: extracted GAPS items');

  // Verify specific content is extracted
  const strJoined = parsed['STRENGTHS'].join(' ');
  ok(strJoined.toLowerCase().includes('react'),            'parseStructured: STRENGTHS contains React');

  const gapJoined = parsed['GAPS'].join(' ');
  ok(gapJoined.toLowerCase().includes('graphql'),         'parseStructured: GAPS contains GraphQL');

  // RECOMMENDATION section (may be single or multi-item)
  ok(parsed['RECOMMENDATION'] !== undefined,               'parseStructured: RECOMMENDATION present');

  // Missing section → empty array
  const missingParsed = llm.parseStructured(sampleText, ['NONEXISTENT_SECTION']);
  ok(Array.isArray(missingParsed['NONEXISTENT_SECTION']),  'parseStructured: missing section → []');
  equal(missingParsed['NONEXISTENT_SECTION'].length, 0,   'parseStructured: missing section → empty');

  // Null/empty input → empty result
  const emptyParsed = llm.parseStructured('', ['STRENGTHS']);
  ok(Array.isArray(emptyParsed['STRENGTHS']),              'parseStructured: empty text → []');
  equal(emptyParsed['STRENGTHS'].length, 0,                'parseStructured: empty text → empty');

  const nullParsed = llm.parseStructured(null, ['STRENGTHS']);
  equal(nullParsed['STRENGTHS'].length, 0,                 'parseStructured: null text → empty');

  // Bold headers are also parsed
  const boldText = `**STRENGTHS**:\n- Expert in Node.js\n\n**GAPS**:\n- No Rust experience\n\nRECOMMENDATION:\nPursue.`;
  const boldParsed = llm.parseStructured(boldText, ['STRENGTHS', 'GAPS']);
  ok(boldParsed['STRENGTHS'].length >= 1,                  'parseStructured: bold STRENGTHS parsed');
  ok(boldParsed['GAPS'].length >= 1,                       'parseStructured: bold GAPS parsed');

  // ── buildMatchPrompt ──────────────────────────────────────────────────────

  const matchCtx = {
    practitionerName:       'Alice Smith',
    practitionerBand:       '8',
    practitionerRole:       'Senior Cloud Engineer',
    practitionerSkills:     ['AWS', 'Kubernetes', 'Terraform'],
    practitionerCerts:      ['AWS Solutions Architect'],
    practitionerSecondaryJRS: 'Platform Engineering',
    seatTitle:              'Cloud Architect',
    seatClient:             'IBM Internal',
    seatProject:            'Project Atlas',
    seatBandLow:            '7B',
    seatBandHigh:           '9',
    seatRequiredSkills:     ['AWS', 'Kubernetes', 'Terraform'],
    seatNiceSkills:         ['GCP', 'Pulumi'],
    keywordScore:           85,
    semanticScore:          72,
  };

  const matchMsgs = llm.buildMatchPrompt(matchCtx);
  ok(Array.isArray(matchMsgs),                             'buildMatchPrompt: returns array');
  equal(matchMsgs.length, 2,                               'buildMatchPrompt: 2 messages');
  equal(matchMsgs[0].role, 'system',                       'buildMatchPrompt: first message is system');
  equal(matchMsgs[1].role, 'user',                         'buildMatchPrompt: second message is user');

  const sysContent = matchMsgs[0].content;
  ok(typeof sysContent === 'string' && sysContent.length > 50, 'buildMatchPrompt: system content non-empty');
  ok(sysContent.includes('STRENGTHS'),                     'buildMatchPrompt: system mentions STRENGTHS');
  ok(sysContent.includes('GAPS'),                          'buildMatchPrompt: system mentions GAPS');
  ok(sysContent.includes('RECOMMENDATION'),                'buildMatchPrompt: system mentions RECOMMENDATION');

  const userContent = matchMsgs[1].content;
  ok(userContent.includes('Alice Smith'),                  'buildMatchPrompt: user content has name');
  ok(userContent.includes('Cloud Architect'),              'buildMatchPrompt: user content has seat title');
  ok(userContent.includes('85/100'),                       'buildMatchPrompt: user content has keyword score');
  ok(userContent.includes('72/100'),                       'buildMatchPrompt: user content has semantic score');
  ok(userContent.includes('AWS'),                          'buildMatchPrompt: user content has required skill');

  // Empty context → should still return 2 messages without throwing
  const emptyMatchMsgs = llm.buildMatchPrompt({});
  equal(emptyMatchMsgs.length, 2,                          'buildMatchPrompt: empty context → 2 messages');
  ok(typeof emptyMatchMsgs[0].content === 'string',        'buildMatchPrompt: empty context → string content');

  // ── buildAssessPrompt ─────────────────────────────────────────────────────

  const assessCtx = {
    cvExcerpt:     'I am an experienced cloud engineer with 8 years in AWS and Kubernetes. Led migration of 50+ microservices.',
    jobRole:       'Senior Cloud Engineer',
    suggestedBand: '8',
    keywordScore:  78,
    semanticScore: 68,
    keywordMatched: ['AWS', 'Kubernetes', 'microservices'],
    keywordGaps:   ['Terraform', 'CI/CD pipelines'],
    demandCount:   12,
  };

  const assessMsgs = llm.buildAssessPrompt(assessCtx);
  ok(Array.isArray(assessMsgs),                            'buildAssessPrompt: returns array');
  equal(assessMsgs.length, 2,                              'buildAssessPrompt: 2 messages');
  equal(assessMsgs[0].role, 'system',                      'buildAssessPrompt: first message is system');
  equal(assessMsgs[1].role, 'user',                        'buildAssessPrompt: second message is user');

  const assessSys = assessMsgs[0].content;
  ok(assessSys.includes('STRENGTHS'),                      'buildAssessPrompt: system mentions STRENGTHS');
  ok(assessSys.includes('RECOMMENDATION'),                 'buildAssessPrompt: system mentions RECOMMENDATION');

  const assessUser = assessMsgs[1].content;
  ok(assessUser.includes('Senior Cloud Engineer'),         'buildAssessPrompt: user has job role');
  ok(assessUser.includes('78/100'),                        'buildAssessPrompt: user has keyword score');
  ok(assessUser.includes('Terraform'),                     'buildAssessPrompt: user has gaps');
  ok(assessUser.includes('12'),                            'buildAssessPrompt: user has demand count');
  ok(assessUser.includes('microservices'),                 'buildAssessPrompt: CV excerpt included');

  // CV excerpt is limited to 2500 chars
  const longCv = 'x'.repeat(5000);
  const longMsgs = llm.buildAssessPrompt({ cvExcerpt: longCv });
  ok(longMsgs[1].content.length < longCv.length,          'buildAssessPrompt: CV excerpt truncated');

  // Empty context → still returns 2 messages
  const emptyAssessMsgs = llm.buildAssessPrompt({});
  equal(emptyAssessMsgs.length, 2,                         'buildAssessPrompt: empty context → 2 messages');

};
