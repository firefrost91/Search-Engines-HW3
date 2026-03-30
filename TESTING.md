# Testing QryEval

## Option 1: Local testing (fastest)

Use the files in **TEST_DIR** and the **trec_eval** binary in **INPUT_DIR**.

### One-time setup

From the **QryEval** directory:

```bash
chmod +x run_test.sh
```

### Commands

| Goal | Command |
|------|--------|
| Run trec_eval on all existing `OUTPUT_DIR/*.teIn` and write `OUTPUT_DIR/*.teOut` | `./run_test.sh` |
| First run QryEval for each test (to create `.teIn`), then run trec_eval | `./run_test.sh --run` |
| Run trec_eval, then diff your `.teOut` vs reference in TEST_DIR | `./run_test.sh --diff` |
| Full run + trec_eval + diff | `./run_test.sh --run --diff` |
| Run only for specific cases (e.g. 0, 5, 16) | `./run_test.sh 0 5 16` |

### Manual single test

```bash
./INPUT_DIR/trec_eval-9.0.4 -m num_q -m num_ret -m num_rel -m num_rel_ret -m map -m recip_rank -m P -m ndcg -m ndcg_cut -m recall.100,500,1000 INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed OUTPUT_DIR/HW1-Train-0.teIn > OUTPUT_DIR/HW1-Train-0.teOut
```

- **Qrel:** `INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed`
- **Your run file:** `OUTPUT_DIR/<test case>.teIn` (produced by QryEval)
- **Store output in:** `OUTPUT_DIR/<test case>.teOut`
- **Reference results:** `TEST_DIR/*.teOut` (compare with yours)

---

## Option 2: trec_eval on your machine

Mac, Linux, and Windows **trec_eval** executables are available. You can also build from source: [trec_eval on GitHub](https://github.com/usnistgov/trec_eval). Put the executable in `INPUT_DIR/` (e.g. `INPUT_DIR/trec_eval-9.0.4`) and use the commands above.

---

## Option 3: Submit / autograder

Use the course’s submission or autograder if required.
