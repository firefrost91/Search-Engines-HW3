/**
 * Block.java
 *
 * Author: Student Name
 * Email: student@andrew.cmu.edu
 *
 * Represents a single block in the blockchain.
 * Each block contains an index (position), timestamp (creation time),
 * transaction data, the previous block's hash, a nonce found by
 * proof-of-work, and the required difficulty.
 *
 * The proof-of-work algorithm searches for a nonce such that the
 * SHA-256 hash of the block's fields starts with 'difficulty' leading
 * zero hex characters (i.e., difficulty zeros in the hex string).
 *
 * Note: Code assistance provided by Claude AI (claude-sonnet-4-6).
 */

import java.security.MessageDigest;
import java.sql.Timestamp;

public class Block {

    // Position of this block in the chain (0-indexed, 0 = genesis)
    private int index;

    // The time this block was created
    private Timestamp timestamp;

    // The transaction data stored in this block (e.g., "Alice pays Bob 100 DSCoin")
    private String data;

    // The hash of the previous block in the chain (empty string for genesis)
    private String previousHash;

    // The nonce found during proof-of-work that satisfies the difficulty
    private int nonce;

    // Number of leading hex zeros required in a valid hash
    private int difficulty;

    /**
     * Constructs a new Block.
     *
     * @param index      the 0-based position of this block in the blockchain
     * @param timestamp  the creation timestamp of this block
     * @param data       the transaction data (e.g., "Alice pays Bob 100 DSCoin")
     * @param difficulty the number of leading zeros required in the hash
     */
    public Block(int index, Timestamp timestamp, String data, int difficulty) {
        this.index = index;
        this.timestamp = timestamp;
        this.data = data;
        this.difficulty = difficulty;
        this.previousHash = "";
        this.nonce = 0;
    }

    /**
     * Computes the SHA-256 hash of this block's contents.
     * The input string is a concatenation of: index, timestamp, data,
     * previousHash, nonce, and difficulty. The result is an uppercase
     * hexadecimal string.
     *
     * @return the uppercase hex SHA-256 hash of this block
     */
    public String computeHash() {
        // Build the input string from all block fields
        String input = index + timestamp.toString() + data + previousHash + nonce + difficulty;
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = md.digest(input.getBytes("UTF-8"));
            // Convert byte array to uppercase hex string
            StringBuilder sb = new StringBuilder();
            for (byte b : hashBytes) {
                sb.append(String.format("%02X", b));
            }
            return sb.toString();
        } catch (Exception e) {
            throw new RuntimeException("SHA-256 computation failed: " + e.getMessage());
        }
    }

    /**
     * Performs the proof-of-work algorithm to find a nonce such that
     * this block's hash starts with 'difficulty' leading zeros (in hex).
     * The nonce is incremented from 0 until a valid hash is found.
     *
     * @return the valid hash string that satisfies the difficulty requirement
     */
    public String proofOfWork() {
        String target = "0".repeat(difficulty); // e.g., "0000" for difficulty 4
        nonce = 0;
        String hash;
        do {
            nonce++;
            hash = computeHash();
        } while (!hash.startsWith(target));
        return hash;
    }

    /**
     * Returns a JSON representation of this block.
     * Field names match those required by the autograder:
     * "index", "timestamp", "Tx", "PrevHash", "nonce", "difficulty"
     *
     * @return JSON string representing this block
     */
    @Override
    public String toString() {
        return "{\"index\" : " + index
                + ",\"timestamp \" : \"" + timestamp + "\""
                + ",\"Tx \" : \"" + data + "\""
                + ",\"PrevHash\" : \"" + previousHash + "\""
                + ",\"nonce\" : " + nonce
                + ",\"difficulty\": " + difficulty + "}";
    }

    // ===================== Getters and Setters =====================

    public int getIndex() { return index; }
    public void setIndex(int index) { this.index = index; }

    public Timestamp getTimestamp() { return timestamp; }
    public void setTimestamp(Timestamp timestamp) { this.timestamp = timestamp; }

    public String getData() { return data; }
    public void setData(String data) { this.data = data; }

    public String getPreviousHash() { return previousHash; }
    public void setPreviousHash(String previousHash) { this.previousHash = previousHash; }

    public int getNonce() { return nonce; }
    public void setNonce(int nonce) { this.nonce = nonce; }

    public int getDifficulty() { return difficulty; }
    public void setDifficulty(int difficulty) { this.difficulty = difficulty; }
}
