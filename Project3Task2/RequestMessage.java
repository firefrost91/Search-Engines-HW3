/**
 * RequestMessage.java  (Task 2 — RSA Digital Signatures version)
 *
 * Author: Student Name
 * Email: student@andrew.cmu.edu
 *
 * Extends the Task 1 RequestMessage with RSA signature fields.
 * The server uses these fields to:
 *   1. Verify that the clientID matches the hash of the public key.
 *   2. Verify that the signature is a valid RSA signature over the
 *      concatenation of (operation + index + difficulty + data).
 *
 * What gets signed (getMessageToSign()):
 *   operation + index + difficulty + data
 *   Example: "add" + "-1" + "4" + "Alice pays Bob 100 DSCoin"
 *            = "add-14Alice pays Bob 100 DSCoin"
 *
 * The signature fields (clientID, publicKeyE, publicKeyN, signature)
 * are NOT included in the signed content.
 *
 * Note: Code assistance provided by Claude AI (claude-sonnet-4-6).
 */
public class RequestMessage {

    // ---- Task 1 fields ----

    /** The operation to perform (e.g., "add", "verify"). */
    private String operation;

    /** Block index for "corrupt"; -1 if not applicable. */
    private int index;

    /** Mining difficulty for "add"; "" if not applicable. */
    private String difficulty;

    /** Transaction data for "add" / new data for "corrupt"; "" if not applicable. */
    private String data;

    // ---- Task 2 signature fields ----

    /**
     * The client's ID: the last 20 bytes (hex-encoded) of SHA-256(e + n).
     * Sent in the clear so the server can verify key ownership.
     */
    private String clientID;

    /** RSA public key exponent e (as decimal string). */
    private String publicKeyE;

    /** RSA public key modulus n (as decimal string). */
    private String publicKeyN;

    /**
     * The RSA signature: encrypt(SHA-256(messageToSign), d, n).
     * The hash has a zero-byte prepended to ensure a positive BigInteger.
     */
    private String signature;

    /** Default constructor required for Gson deserialization. */
    public RequestMessage() {
        this.index = -1;
        this.difficulty = "";
        this.data = "";
    }

    /**
     * Constructor for operations that need no additional parameters.
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
     * Constructor for "corrupt" (needs block index and new data).
     *
     * @param operation the operation string ("corrupt")
     * @param index     the block index to corrupt
     * @param data      the new (corrupted) data
     */
    public RequestMessage(String operation, int index, String data) {
        this.operation = operation;
        this.index = index;
        this.difficulty = "";
        this.data = data;
    }

    /**
     * Constructor for "add" (needs difficulty and transaction data).
     *
     * @param operation  the operation string ("add")
     * @param difficulty the mining difficulty
     * @param data       the transaction string
     */
    public RequestMessage(String operation, String difficulty, String data) {
        this.operation = operation;
        this.difficulty = difficulty;
        this.data = data;
        this.index = -1;
    }

    /**
     * Returns the content that should be signed / verified.
     * This is the concatenation of the four payload fields (NOT the
     * signature fields themselves, and NOT the JSON key names).
     *
     * @return the string to be hashed and signed
     */
    public String getMessageToSign() {
        return operation + index + difficulty + data;
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

    public String getClientID() { return clientID; }
    public void setClientID(String clientID) { this.clientID = clientID; }

    public String getPublicKeyE() { return publicKeyE; }
    public void setPublicKeyE(String publicKeyE) { this.publicKeyE = publicKeyE; }

    public String getPublicKeyN() { return publicKeyN; }
    public void setPublicKeyN(String publicKeyN) { this.publicKeyN = publicKeyN; }

    public String getSignature() { return signature; }
    public void setSignature(String signature) { this.signature = signature; }

    @Override
    public String toString() {
        return "RequestMessage{"
                + "operation='" + operation + '\''
                + ", index=" + index
                + ", difficulty='" + difficulty + '\''
                + ", data='" + data + '\''
                + ", clientID='" + clientID + '\''
                + ", publicKeyE='" + publicKeyE + '\''
                + ", publicKeyN='" + publicKeyN + '\''
                + ", signature='" + signature + '\''
                + '}';
    }
}
