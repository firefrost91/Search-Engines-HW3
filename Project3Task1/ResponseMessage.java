/**
 * ResponseMessage.java
 *
 * Author: Student Name
 * Email: student@andrew.cmu.edu
 *
 * POJO that encapsulates all server-to-client responses for the Task 1
 * (and Task 2) blockchain application. Instances are serialized to JSON
 * (via Gson) before being sent over a TCP socket, and deserialized on
 * the client side.
 *
 * Fields:
 *   status        - "success" or "error"
 *   message       - Human-readable description of the result
 *   data          - Additional payload (e.g., JSON string for "view"/"status")
 *   executionTime - Time in milliseconds taken to execute the operation
 *
 * Note: Code assistance provided by Claude AI (claude-sonnet-4-6).
 */
public class ResponseMessage {

    // "success" if the operation completed normally; "error" otherwise
    private String status;

    // Human-readable result message (e.g., timing info, verification result)
    private String message;

    // Additional data payload (JSON string for complex responses; "" otherwise)
    private String data;

    // Wall-clock time in milliseconds for the server-side operation
    private long executionTime;

    /** Default constructor required for Gson deserialization. */
    public ResponseMessage() {
    }

    /**
     * Constructor for simple success/error responses with no extra data.
     *
     * @param status  "success" or "error"
     * @param message descriptive message
     */
    public ResponseMessage(String status, String message) {
        this.status = status;
        this.message = message;
        this.data = "";
        this.executionTime = 0;
    }

    /**
     * Full constructor.
     *
     * @param status        "success" or "error"
     * @param message       descriptive message
     * @param data          JSON payload string (or "" if none)
     * @param executionTime milliseconds taken on server side
     */
    public ResponseMessage(String status, String message, String data, long executionTime) {
        this.status = status;
        this.message = message;
        this.data = data;
        this.executionTime = executionTime;
    }

    // ===================== Getters and Setters =====================

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }

    public String getData() { return data; }
    public void setData(String data) { this.data = data; }

    public long getExecutionTime() { return executionTime; }
    public void setExecutionTime(long executionTime) { this.executionTime = executionTime; }

    @Override
    public String toString() {
        return "ResponseMessage{"
                + "status='" + status + '\''
                + ", message='" + message + '\''
                + ", data='" + data + '\''
                + ", executionTime=" + executionTime
                + '}';
    }
}
