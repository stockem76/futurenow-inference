'use strict';
/**
 * run-all.js — @futurenow/inference test runner.
 *
 * Usage:  node test/run-all.js
 *
 * Runs every *.test.js file found in this directory sequentially and
 * prints a pass/fail summary.  No third-party test framework required.
 */

const fs   = require('fs');
const path = require('path');

const TEST_DIR = __dirname;
const RED   = '\x1b[31m';
const GREEN = '\x1b[32m';
const CYAN  = '\x1b[36m';
const BOLD  = '\x1b[1m';
const RESET = '\x1b[0m';

let totalPass = 0;
let totalFail = 0;

// ── Minimal assertion helpers (shared via global) ─────────────────────────────

global._assert = {
  /**
   * @param {boolean} condition
   * @param {string}  label
   */
  ok(condition, label) {
    if (condition) {
      console.log(`  ${GREEN}✓${RESET} ${label}`);
      totalPass++;
    } else {
      console.log(`  ${RED}✗ FAIL${RESET} ${label}`);
      totalFail++;
    }
  },

  /**
   * @param {*}      actual
   * @param {*}      expected
   * @param {string} label
   */
  equal(actual, expected, label) {
    const pass = actual === expected;
    if (pass) {
      console.log(`  ${GREEN}✓${RESET} ${label}`);
      totalPass++;
    } else {
      console.log(`  ${RED}✗ FAIL${RESET} ${label}  (got ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)})`);
      totalFail++;
    }
  },

  /**
   * Assert that `actual` is within `delta` of `expected`.
   * @param {number} actual
   * @param {number} expected
   * @param {number} delta
   * @param {string} label
   */
  near(actual, expected, delta, label) {
    const pass = Math.abs(actual - expected) <= delta;
    if (pass) {
      console.log(`  ${GREEN}✓${RESET} ${label}`);
      totalPass++;
    } else {
      console.log(`  ${RED}✗ FAIL${RESET} ${label}  (got ${actual}, expected ${expected} ±${delta})`);
      totalFail++;
    }
  },
};

// ── Discover and run test files ───────────────────────────────────────────────

const testFiles = fs.readdirSync(TEST_DIR)
  .filter(f => f.endsWith('.test.js'))
  .sort();

(async () => {
  for (const file of testFiles) {
    console.log(`\n${CYAN}${BOLD}▶ ${file}${RESET}`);
    try {
      // Each test file can export an async function or just run synchronously
      const mod = require(path.join(TEST_DIR, file));
      if (typeof mod === 'function') await mod();
    } catch (err) {
      console.log(`  ${RED}✗ UNCAUGHT ERROR${RESET} ${err.message}`);
      console.error(err);
      totalFail++;
    }
  }

  // ── Summary ─────────────────────────────────────────────────────────────────
  const total = totalPass + totalFail;
  console.log(`\n${BOLD}────────────────────────────────────────${RESET}`);
  if (totalFail === 0) {
    console.log(`${GREEN}${BOLD}All ${total} tests passed.${RESET}`);
    process.exit(0);
  } else {
    console.log(`${RED}${BOLD}${totalFail} / ${total} tests FAILED.${RESET}`);
    process.exit(1);
  }
})();
