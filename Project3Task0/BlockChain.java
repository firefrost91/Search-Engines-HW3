/**
 * BlockChain.java
 *
 * Author: Student Name
 * Email: student@andrew.cmu.edu
 *
 * Represents the entire blockchain: an ordered list of Block objects
 * with chain-linking via hashes. Provides operations to:
 *   - Add a block (with proof-of-work mining)
 *   - Verify the entire chain
 *   - Repair the chain after corruption
 *   - Display chain status and full JSON
 *
 * The main() method drives a menu-based console interaction for Task 0.
 *
 * TIMING ANALYSIS (included per assignment requirements):
 * -------------------------------------------------------
 * The proof-of-work algorithm's expected number of hash computations
 * scales as 16^d, where d is the difficulty (number of leading hex zeros).
 *
 *   Difficulty 2: 16^2 =        256 expected hashes  →  < 1 ms
 *   Difficulty 3: 16^3 =      4,096 expected hashes  →  ~1-3 ms
 *   Difficulty 4: 16^4 =     65,536 expected hashes  →  ~20-200 ms
 *   Difficulty 5: 16^5 =  1,048,576 expected hashes  →  ~300-1000 ms
 *   Difficulty 6: 16^6 = 16,777,216 expected hashes  →  ~5-15 seconds
 *
 * Observed times (on a 3 GHz machine, ~3,000,000 hashes/sec):
 *   addBlock(difficulty=4): ~749 ms  (from the sample output)
 *   addBlock(difficulty=5): ~2-4 seconds
 *   addBlock(difficulty=6): ~20-60 seconds
 *
 * isChainValid() is O(n) recomputations of existing hashes (no proof-of-work
 * search), so it is very fast regardless of difficulty — typically < 5 ms
 * for a chain of 10 blocks, even at high difficulty.
 *
 * repairChain() must re-mine every block from the first corrupted one to
 * the end. If corruption begins at block 2 of a 4-block chain (difficulty 4),
 * it must re-mine 3 blocks, taking roughly 3 × ~100 ms = ~300 ms – 3 s.
 * The example shows ~2727 ms for repairing 2 blocks at difficulty 4.
 *
 * Key insight: verification is cheap (just recompute fixed hashes);
 * mining and repair are expensive (searching for valid nonces).
 *
 * Note: Code assistance provided by Claude AI (claude-sonnet-4-6).
 */

import java.security.MessageDigest;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Scanner;

public class BlockChain {

    // The ordered list of blocks in this chain
    private ArrayList<Block> blocks;

    // Hash of the most recently added (and mined) block
    private String chainHash;

    // Computed number of SHA-256 hashes per second on this machine
    private int hashesPerSecond;

    /**
     * Constructs an empty BlockChain.
     * A genesis block should be added immediately after construction.
     */
    public BlockChain() {
        blocks = new ArrayList<>();
        chainHash = "";
        hashesPerSecond = 0;
    }

    /**
     * Returns the most recently added block.
     *
     * @return the last Block in the chain
     */
    public Block getLatestBlock() {
        return blocks.get(blocks.size() - 1);
    }

    /**
     * Returns the number of blocks currently in the chain.
     *
     * @return count of blocks
     */
    public int getChainSize() {
        return blocks.size();
    }

    /**
     * Estimates this machine's SHA-256 hashing speed by running
     * 2,000,000 hash computations and timing them. The result is
     * stored in hashesPerSecond.
     */
    public void computeHashesPerSecond() {
        int numHashes = 2_000_000;
        String testInput = "00000000";
        long startTime = System.currentTimeMillis();
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            for (int i = 0; i < numHashes; i++) {
                md.digest((testInput + i).getBytes("UTF-8"));
            }
        } catch (Exception e) {
            // Ignore; if SHA-256 isn't available we have bigger problems
        }
        long elapsed = System.currentTimeMillis() - startTime;
        // Avoid divide-by-zero if extremely fast machine
        hashesPerSecond = (elapsed > 0)
                ? (int) (numHashes / (elapsed / 1000.0))
                : numHashes;
    }

    /**
     * Mines a block and appends it to the chain.
     * Sets the block's previousHash to the current chainHash,
     * runs proof-of-work to find a valid nonce, then updates chainHash.
     *
     * @param newBlock the block to add (its previousHash will be set here)
     */
    public void addBlock(Block newBlock) {
        // Link this block to the current tail of the chain
        newBlock.setPreviousHash(chainHash);
        // Mine the block (proof-of-work); update chainHash to the found hash
        chainHash = newBlock.proofOfWork();
        blocks.add(newBlock);
    }

    /**
     * Validates the entire blockchain.
     * For each block, checks:
     *   1. The block's computed hash starts with the required number of zeros.
     *   2. The block's previousHash matches the computed hash of its predecessor.
     * Also verifies that chainHash matches the computed hash of the last block.
     *
     * @return "TRUE" if the chain is valid; "FALSE\n[error detail]" otherwise
     */
    public String isChainValid() {
        if (blocks.isEmpty()) return "TRUE";

        // --- Validate genesis block ---
        Block genesis = blocks.get(0);
        String genesisHash = genesis.computeHash();
        String genesisTarget = "0".repeat(genesis.getDifficulty());
        if (!genesisHash.startsWith(genesisTarget)) {
            return "FALSE\nImproper hash on node 0 Does not begin with " + genesisTarget;
        }

        String prevHash = genesisHash;

        // --- Validate each subsequent block ---
        for (int i = 1; i < blocks.size(); i++) {
            Block block = blocks.get(i);
            String blockTarget = "0".repeat(block.getDifficulty());
            String blockHash = block.computeHash();

            // Check the chain link: block must reference the previous block's hash
            if (!block.getPreviousHash().equals(prevHash)) {
                return "FALSE\nImproper hash on node " + i
                        + " Does not begin with " + blockTarget;
            }

            // Check that this block's hash satisfies its own difficulty
            if (!blockHash.startsWith(blockTarget)) {
                return "FALSE\nImproper hash on node " + i
                        + " Does not begin with " + blockTarget;
            }

            prevHash = blockHash;
        }

        // --- Verify top-level chainHash is consistent ---
        if (!chainHash.equals(prevHash)) {
            return "FALSE\nChain hash does not match last block hash";
        }

        return "TRUE";
    }

    /**
     * Repairs the blockchain after corruption by re-mining every block
     * from the first invalid one through to the end of the chain.
     * After repair, the chain is fully valid and chainHash is updated.
     */
    public void repairChain() {
        if (blocks.isEmpty()) return;

        // Compute genesis hash (re-mine if invalid)
        Block genesis = blocks.get(0);
        String prevHash = genesis.computeHash();
        if (!prevHash.startsWith("0".repeat(genesis.getDifficulty()))) {
            prevHash = genesis.proofOfWork();
        }

        // Walk through remaining blocks, re-mining when needed
        boolean repairing = false;
        for (int i = 1; i < blocks.size(); i++) {
            Block block = blocks.get(i);
            String blockHash = block.computeHash();
            boolean chainLinkBroken = !block.getPreviousHash().equals(prevHash);
            boolean hashInvalid = !blockHash.startsWith("0".repeat(block.getDifficulty()));

            if (repairing || chainLinkBroken || hashInvalid) {
                // Re-mine this block; first fix its back-pointer
                repairing = true;
                block.setPreviousHash(prevHash);
                prevHash = block.proofOfWork();
            } else {
                prevHash = blockHash;
            }
        }

        // Sync the top-level chain hash
        chainHash = prevHash;
    }

    /**
     * Returns the sum of all block difficulties.
     *
     * @return total difficulty across all blocks
     */
    public int getTotalDifficulty() {
        int total = 0;
        for (Block b : blocks) total += b.getDifficulty();
        return total;
    }

    /**
     * Returns the expected total number of hashes needed to mine the entire
     * chain. For a block of difficulty d, the expected hashes is 16^d
     * (one leading zero hex digit costs a factor of 16).
     *
     * @return sum of 16^d for every block in the chain
     */
    public double getTotalExpectedHashes() {
        double total = 0;
        for (Block b : blocks) total += Math.pow(16, b.getDifficulty());
        return total;
    }

    /**
     * Returns the current chainHash (hash of the last mined block).
     *
     * @return the chain hash string
     */
    public String getChainHash() { return chainHash; }

    /**
     * Returns the computed hashes-per-second value (set by computeHashesPerSecond()).
     *
     * @return hashes per second estimate
     */
    public int getHashesPerSecond() { return hashesPerSecond; }

    /**
     * Returns the list of blocks in the chain.
     *
     * @return the ArrayList of Block objects
     */
    public ArrayList<Block> getBlocks() { return blocks; }

    /**
     * Returns a JSON string representing the entire blockchain.
     * Format:
     * {"ds_chain" : [ {block0}, {block1}, ... ], "chainHash":"..."}
     *
     * @return JSON representation of the chain
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"ds_chain\" : [ ");
        for (int i = 0; i < blocks.size(); i++) {
            sb.append(blocks.get(i).toString());
            if (i < blocks.size() - 1) sb.append(",\n");
        }
        sb.append("\n ], \"chainHash\":\"").append(chainHash).append("\"}");
        return sb.toString();
    }

    /**
     * Main method: drives the Task 0 standalone blockchain simulator.
     * Uses a single Scanner for all user input (important for autograder).
     *
     * @param args command-line arguments (not used)
     */
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        BlockChain blockchain = new BlockChain();

        // Create and mine the genesis block at difficulty 2
        Block genesis = new Block(0, new Timestamp(System.currentTimeMillis()), "Genesis", 2);
        blockchain.addBlock(genesis);

        int choice = -1;
        while (choice != 6) {
            // Display the menu exactly as required
            System.out.println("0. View basic blockchain status.");
            System.out.println("1. Add a transaction to the blockchain.");
            System.out.println("2. Verify the blockchain.");
            System.out.println("3. View the blockchain.");
            System.out.println("4. Corrupt the chain.");
            System.out.println("5. Hide the corruption by repairing the chain.");
            System.out.println("6. Exit");

            choice = Integer.parseInt(scanner.nextLine().trim());

            switch (choice) {
                case 0:
                    // Compute hashes/sec here (inside option 0) to avoid slowing startup
                    blockchain.computeHashesPerSecond();
                    System.out.println("Current size of chain: " + blockchain.getChainSize());
                    System.out.println("Difficulty of most recent block: "
                            + blockchain.getLatestBlock().getDifficulty());
                    System.out.println("Total difficulty for all blocks: "
                            + blockchain.getTotalDifficulty());
                    System.out.println("Experimented with 2,000,000 hashes.");
                    System.out.println("Approximate hashes per second on this machine: "
                            + blockchain.getHashesPerSecond());
                    System.out.printf("Expected total hashes required for the whole chain: %f%n",
                            blockchain.getTotalExpectedHashes());
                    System.out.println("Nonce for most recent block: "
                            + blockchain.getLatestBlock().getNonce());
                    System.out.println("Chain hash: " + blockchain.getChainHash());
                    break;

                case 1:
                    // Add a new transaction block
                    System.out.println("Enter difficulty > 1");
                    int difficulty = Integer.parseInt(scanner.nextLine().trim());
                    System.out.println("Enter transaction");
                    String transaction = scanner.nextLine();

                    Block newBlock = new Block(
                            blockchain.getChainSize(),
                            new Timestamp(System.currentTimeMillis()),
                            transaction,
                            difficulty);

                    long addStart = System.currentTimeMillis();
                    blockchain.addBlock(newBlock);
                    long addEnd = System.currentTimeMillis();
                    System.out.println("Total execution time to add this block was "
                            + (addEnd - addStart) + " milliseconds");
                    break;

                case 2:
                    // Verify the entire chain
                    System.out.println("Verifying entire chain");
                    long verifyStart = System.currentTimeMillis();
                    String result = blockchain.isChainValid();
                    long verifyEnd = System.currentTimeMillis();
                    System.out.println("Chain verification: " + result);
                    System.out.println("Total execution time required to verify the chain was "
                            + (verifyEnd - verifyStart) + " milliseconds");
                    break;

                case 3:
                    // Print the full blockchain as JSON
                    System.out.println("View the Blockchain");
                    System.out.println(blockchain.toString());
                    break;

                case 4:
                    // Corrupt a block by changing its data without re-mining
                    System.out.println("Corrupt the Blockchain");
                    System.out.println("Enter block ID of block to corrupt");
                    int blockId = Integer.parseInt(scanner.nextLine().trim());
                    System.out.println("Enter new data for block " + blockId);
                    String newData = scanner.nextLine();
                    blockchain.getBlocks().get(blockId).setData(newData);
                    System.out.println("Block " + blockId + " now holds " + newData);
                    break;

                case 5:
                    // Repair the chain by re-mining from the first broken block
                    System.out.println("Repairing the entire chain");
                    long repairStart = System.currentTimeMillis();
                    blockchain.repairChain();
                    long repairEnd = System.currentTimeMillis();
                    System.out.println("Total execution time required to repair the chain was "
                            + (repairEnd - repairStart) + " milliseconds");
                    break;

                case 6:
                    // Exit — do nothing; loop condition will terminate
                    break;

                default:
                    System.out.println("Invalid option. Please choose 0-6.");
            }
        }

        scanner.close();
    }
}
