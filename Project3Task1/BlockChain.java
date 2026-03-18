/**
 * BlockChain.java
 *
 * Author: Student Name
 * Email: student@andrew.cmu.edu
 *
 * Represents the blockchain used by the Task 1 server.
 * Provides operations to add blocks, verify the chain, repair it,
 * and serialize to JSON. No menu/main here — the server drives interaction.
 *
 * Note: Code assistance provided by Claude AI (claude-sonnet-4-6).
 */

import java.security.MessageDigest;
import java.sql.Timestamp;
import java.util.ArrayList;

public class BlockChain {

    // Ordered list of blocks
    private ArrayList<Block> blocks;

    // Hash of the most recently added (mined) block
    private String chainHash;

    // Hashes per second measured on this machine
    private int hashesPerSecond;

    public BlockChain() {
        blocks = new ArrayList<>();
        chainHash = "";
        hashesPerSecond = 0;
    }

    /** Returns the most recently added block. */
    public Block getLatestBlock() {
        return blocks.get(blocks.size() - 1);
    }

    /** Returns the number of blocks in the chain. */
    public int getChainSize() {
        return blocks.size();
    }

    /**
     * Estimates this machine's SHA-256 hashing speed by running
     * 2,000,000 hash computations and timing the result.
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
            // ignore
        }
        long elapsed = System.currentTimeMillis() - startTime;
        hashesPerSecond = (elapsed > 0)
                ? (int) (numHashes / (elapsed / 1000.0))
                : numHashes;
    }

    /**
     * Mines a block and appends it to the chain.
     * Sets the block's previousHash to the current chainHash,
     * runs proof-of-work, then updates chainHash.
     *
     * @param newBlock the block to add
     */
    public void addBlock(Block newBlock) {
        newBlock.setPreviousHash(chainHash);
        chainHash = newBlock.proofOfWork();
        blocks.add(newBlock);
    }

    /**
     * Validates the entire chain.
     *
     * @return "TRUE" if valid; "FALSE\n[error detail]" otherwise
     */
    public String isChainValid() {
        if (blocks.isEmpty()) return "TRUE";

        Block genesis = blocks.get(0);
        String genesisHash = genesis.computeHash();
        String genesisTarget = "0".repeat(genesis.getDifficulty());
        if (!genesisHash.startsWith(genesisTarget)) {
            return "FALSE\nImproper hash on node 0 Does not begin with " + genesisTarget;
        }

        String prevHash = genesisHash;
        for (int i = 1; i < blocks.size(); i++) {
            Block block = blocks.get(i);
            String blockTarget = "0".repeat(block.getDifficulty());
            String blockHash = block.computeHash();

            if (!block.getPreviousHash().equals(prevHash)) {
                return "FALSE\nImproper hash on node " + i
                        + " Does not begin with " + blockTarget;
            }
            if (!blockHash.startsWith(blockTarget)) {
                return "FALSE\nImproper hash on node " + i
                        + " Does not begin with " + blockTarget;
            }
            prevHash = blockHash;
        }

        if (!chainHash.equals(prevHash)) {
            return "FALSE\nChain hash does not match last block hash";
        }
        return "TRUE";
    }

    /**
     * Repairs the chain by re-mining every block from the first invalid one.
     */
    public void repairChain() {
        if (blocks.isEmpty()) return;

        Block genesis = blocks.get(0);
        String prevHash = genesis.computeHash();
        if (!prevHash.startsWith("0".repeat(genesis.getDifficulty()))) {
            prevHash = genesis.proofOfWork();
        }

        boolean repairing = false;
        for (int i = 1; i < blocks.size(); i++) {
            Block block = blocks.get(i);
            String blockHash = block.computeHash();
            boolean chainLinkBroken = !block.getPreviousHash().equals(prevHash);
            boolean hashInvalid = !blockHash.startsWith("0".repeat(block.getDifficulty()));

            if (repairing || chainLinkBroken || hashInvalid) {
                repairing = true;
                block.setPreviousHash(prevHash);
                prevHash = block.proofOfWork();
            } else {
                prevHash = blockHash;
            }
        }
        chainHash = prevHash;
    }

    /** Returns the sum of all block difficulties. */
    public int getTotalDifficulty() {
        int total = 0;
        for (Block b : blocks) total += b.getDifficulty();
        return total;
    }

    /**
     * Returns the expected total hash count: sum of 16^d for each block
     * (each leading hex zero costs a factor of 16 in expected hashes).
     */
    public double getTotalExpectedHashes() {
        double total = 0;
        for (Block b : blocks) total += Math.pow(16, b.getDifficulty());
        return total;
    }

    public String getChainHash() { return chainHash; }
    public int getHashesPerSecond() { return hashesPerSecond; }
    public ArrayList<Block> getBlocks() { return blocks; }

    /**
     * Returns a JSON string of the entire blockchain.
     * Format: {"ds_chain" : [ ... ], "chainHash":"..."}
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
}
