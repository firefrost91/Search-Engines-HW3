/**
 * ServerTCP.java
 *
 * Author: Student Name
 * Email: student@andrew.cmu.edu
 *
 * Task 1 blockchain server. Listens on port 7777, accepts one client at a
 * time (runs a persistent loop for multiple requests from the same client),
 * and processes JSON-encoded RequestMessage objects, returning JSON-encoded
 * ResponseMessage objects.
 *
 * Supported operations:
 *   "status"  - Returns blockchain metrics as JSON in response data field
 *   "add"     - Mines a new block; returns timing in message
 *   "verify"  - Validates chain; returns TRUE/FALSE in message
 *   "view"    - Returns the full blockchain JSON in data field
 *   "corrupt" - Corrupts a block's data without re-mining
 *   "repair"  - Re-mines from first corrupted block
 *
 * Server console output per request:
 *   1. "We have a visitor"
 *   2. The incoming JSON request
 *   3. The outgoing JSON response
 *   4. "Number of Blocks on Chain == N"
 *
 * Note: Code assistance provided by Claude AI (claude-sonnet-4-6).
 */

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.sql.Timestamp;
import java.util.Scanner;

public class ServerTCP {

    // Port the server listens on
    private static final int PORT = 7777;

    // The blockchain lives on the server; it persists across client reconnects
    private static BlockChain blockchain = new BlockChain();

    // Gson instance for JSON serialization / deserialization
    private static final Gson gson = new Gson();

    public static void main(String[] args) {
        // Initialize the blockchain with a genesis block
        Block genesis = new Block(0, new Timestamp(System.currentTimeMillis()), "Genesis", 2);
        blockchain.addBlock(genesis);

        System.out.println("Blockchain server running on port " + PORT);

        try (ServerSocket listenSocket = new ServerSocket(PORT)) {
            // Accept connections repeatedly; each client may send many requests
            while (true) {
                Socket clientSocket = listenSocket.accept();
                handleClient(clientSocket);
            }
        } catch (IOException e) {
            System.out.println("Server IO error: " + e.getMessage());
        }
    }

    /**
     * Handles all requests from a single connected client.
     * Reads one JSON request per line, processes it, and sends back one
     * JSON response per line. Loops until the client disconnects or
     * an "exit" operation is received.
     *
     * @param clientSocket the accepted client socket
     */
    private static void handleClient(Socket clientSocket) {
        try (
            Scanner in = new Scanner(clientSocket.getInputStream());
            PrintWriter out = new PrintWriter(
                new BufferedWriter(new OutputStreamWriter(clientSocket.getOutputStream())))
        ) {
            while (in.hasNextLine()) {
                String jsonRequest = in.nextLine();
                System.out.println("We have a visitor");
                System.out.println(jsonRequest);

                // Deserialize the request
                RequestMessage request = gson.fromJson(jsonRequest, RequestMessage.class);
                String operation = request.getOperation();

                ResponseMessage response;

                // Route to the appropriate handler
                switch (operation) {
                    case "status":
                        response = handleStatus();
                        break;
                    case "add":
                        response = handleAdd(request);
                        break;
                    case "verify":
                        response = handleVerify();
                        break;
                    case "view":
                        response = handleView();
                        break;
                    case "corrupt":
                        response = handleCorrupt(request);
                        break;
                    case "repair":
                        response = handleRepair();
                        break;
                    case "exit":
                        // Client is exiting; send acknowledgement and stop loop
                        response = new ResponseMessage("success", "Server acknowledged exit");
                        String exitJson = gson.toJson(response);
                        System.out.println(exitJson);
                        out.println(exitJson);
                        out.flush();
                        System.out.println("Number of Blocks on Chain == " + blockchain.getChainSize());
                        return;
                    default:
                        response = new ResponseMessage("error", "Unknown operation: " + operation);
                }

                String jsonResponse = gson.toJson(response);
                System.out.println(jsonResponse);
                out.println(jsonResponse);
                out.flush();
                System.out.println("Number of Blocks on Chain == " + blockchain.getChainSize());
            }
        } catch (IOException e) {
            System.out.println("Client connection error: " + e.getMessage());
        }
    }

    // ===================== Operation Handlers =====================

    /**
     * Handles the "status" operation.
     * Computes hashes/sec and returns blockchain metrics as a JSON string
     * in the response's data field.
     *
     * @return ResponseMessage with blockchain status data
     */
    private static ResponseMessage handleStatus() {
        blockchain.computeHashesPerSecond();

        // Build status JSON manually to control field names exactly
        JsonObject statusJson = new JsonObject();
        statusJson.addProperty("currentSizeOfChain", blockchain.getChainSize());
        statusJson.addProperty("difficultyOfMostRecentBlock",
                blockchain.getLatestBlock().getDifficulty());
        statusJson.addProperty("totalDifficultyForAllBlocks", blockchain.getTotalDifficulty());
        statusJson.addProperty("experimentedHashes", 2_000_000);
        statusJson.addProperty("approximateHashesPerSecond", blockchain.getHashesPerSecond());
        statusJson.addProperty("expectedTotalHashesRequired",
                blockchain.getTotalExpectedHashes());
        statusJson.addProperty("nonceForMostRecentBlock",
                String.valueOf(blockchain.getLatestBlock().getNonce()));
        statusJson.addProperty("chainHash", blockchain.getChainHash());

        return new ResponseMessage("success", "Blockchain status retrieved",
                gson.toJson(statusJson), 0);
    }

    /**
     * Handles the "add" operation: mines a new block with the requested
     * difficulty and data, then reports the elapsed time.
     *
     * @param request the deserialized client request
     * @return ResponseMessage with mining time in the message field
     */
    private static ResponseMessage handleAdd(RequestMessage request) {
        int difficulty = Integer.parseInt(request.getDifficulty());
        String data = request.getData();

        Block newBlock = new Block(
                blockchain.getChainSize(),
                new Timestamp(System.currentTimeMillis()),
                data,
                difficulty);

        long start = System.currentTimeMillis();
        blockchain.addBlock(newBlock);
        long elapsed = System.currentTimeMillis() - start;

        return new ResponseMessage(
                "success",
                "Total execution time to add this block was " + elapsed + " milliseconds",
                "",
                0);
    }

    /**
     * Handles the "verify" operation: validates the entire chain and
     * returns TRUE or FALSE with timing information.
     *
     * @return ResponseMessage with verification result and elapsed time
     */
    private static ResponseMessage handleVerify() {
        long start = System.currentTimeMillis();
        String result = blockchain.isChainValid();
        long elapsed = System.currentTimeMillis() - start;

        return new ResponseMessage(
                "success",
                "Chain verification: " + result
                        + "\nTotal execution time required to verify the chain was "
                        + elapsed + " milliseconds",
                "",
                elapsed);
    }

    /**
     * Handles the "view" operation: serializes the blockchain to JSON and
     * returns it in the response's data field.
     *
     * @return ResponseMessage containing the blockchain JSON string
     */
    private static ResponseMessage handleView() {
        return new ResponseMessage("success", "Blockchain retrieved",
                blockchain.toString(), 0);
    }

    /**
     * Handles the "corrupt" operation: changes a block's data without
     * re-mining (intentionally breaks the chain for demonstration).
     *
     * @param request the deserialized client request (must include index and data)
     * @return ResponseMessage confirming the corruption
     */
    private static ResponseMessage handleCorrupt(RequestMessage request) {
        int blockId = request.getIndex();
        String newData = request.getData();
        blockchain.getBlocks().get(blockId).setData(newData);
        return new ResponseMessage(
                "success",
                "Block " + blockId + " now holds " + newData);
    }

    /**
     * Handles the "repair" operation: re-mines the chain from the first
     * broken block and reports elapsed time.
     *
     * @return ResponseMessage with repair timing
     */
    private static ResponseMessage handleRepair() {
        long start = System.currentTimeMillis();
        blockchain.repairChain();
        long elapsed = System.currentTimeMillis() - start;

        return new ResponseMessage(
                "success",
                "Total execution time required to repair the chain was " + elapsed + " milliseconds",
                "",
                elapsed);
    }
}
