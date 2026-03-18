/**
 * RequestMessage.java
 *
 * Author: Student Name
 * Email: student@andrew.cmu.edu
 *
 * POJO (Plain Old Java Object) that encapsulates all client-to-server
 * requests for the Task 1 blockchain application. Instances are serialized
 * to JSON (via Gson) before being sent over a TCP socket, and deserialized
 * from JSON on the server side.
 *
 * Operation strings (must match exactly):
 *   "status"  - View basic blockchain status
 *   "add"     - Add a transaction block
 *   "verify"  - Verify the chain
 *   "view"    - View full blockchain JSON
 *   "corrupt" - Corrupt a specific block
 *   "repair"  - Repair the chain
 *
 * Note: Code assistance provided by Claude AI (claude-sonnet-4-6).
 */
public class RequestMessage {

    // The operation to perform on the server (e.g., "add", "verify")
    private String operation;

    // Block index for "corrupt" operation; -1 if not applicable
    private int index;

    // Mining difficulty for "add" operation; "" if not applicable
    private String difficulty;

    // Transaction data for "add" / new data for "corrupt"; "" if not applicable
    private String data;

    /** Default constructor required for Gson deserialization. */
    public RequestMessage() {
        this.index = -1;
        this.difficulty = "";
        this.data = "";
    }

    /**
     * Constructor for operations that need no additional parameters
     * (e.g., "status", "verify", "view", "repair").
     *
     * @param operation the operation string
     */
    public RequestMessage(String operation) {
        this.operation = operation;
        this.index = -1;
        this.difficulty = "";
        this.data = "";
    }

    /**
     * Constructor for the "corrupt" operation, which requires a block index
     * and replacement data.
     *
     * @param operation the operation string ("corrupt")
     * @param index     the index of the block to corrupt
     * @param data      the new (corrupted) data for that block
     */
    public RequestMessage(String operation, int index, String data) {
        this.operation = operation;
        this.index = index;
        this.difficulty = "";
        this.data = data;
    }

    /**
     * Constructor for the "add" operation, which requires a difficulty level
     * and transaction data.
     *
     * @param operation  the operation string ("add")
     * @param difficulty the desired mining difficulty
     * @param data       the transaction string
     */
    public RequestMessage(String operation, String difficulty, String data) {
        this.operation = operation;
        this.difficulty = difficulty;
        this.data = data;
        this.index = -1;
    }

    // ===================== Getters and Setters =====================

    public String getOperation() { return operation; }
    public void setOperation(String operation) { this.operation = operation; }

    public int getIndex() { return index; }
    public void setIndex(int index) { this.index = index; }

    public String getDifficulty() { return difficulty; }
    public void setDifficulty(String difficulty) { this.difficulty = difficulty; }

    public String getData() { return data; }
    public void setData(String data) { this.data = data; }

    @Override
    public String toString() {
        return "RequestMessage{"
                + "operation='" + operation + '\''
                + ", index=" + index
                + ", difficulty='" + difficulty + '\''
                + ", data='" + data + '\''
                + '}';
    }
}
