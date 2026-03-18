/**
 * VerifyingServerTCP.java
 *
 * Author: Student Name
 * Email: student@andrew.cmu.edu
 *
 * Task 2 blockchain server with RSA signature verification.
 *
 * For each incoming request the server performs TWO checks before
 * processing any blockchain operation:
 *
 *   Check 1 — ID verification:
 *     Compute SHA-256(publicKeyE + publicKeyN) and compare its last 20 bytes
 *     (hex-encoded) against the clientID field. If they differ, the request
 *     is rejected.
 *
 *   Check 2 — Signature verification:
 *     Rebuild messageToSign = operation + index + difficulty + data.
 *     SHA-256 hash it (with 0x00 prefix byte for positive BigInteger).
 *     RSA-decrypt the signature using (e, n): decrypted = sig^e mod n.
 *     Compare the decrypted value to the computed hash BigInteger.
 *     If they differ, the request is rejected.
 *
 * If both checks pass, the operation is executed exactly as in Task 1.
 * Otherwise, "Error in request" is returned.
 *
 * Server console output per request:
 *   1. "We have a visitor"
 *   2. The incoming JSON request
 *   3. Client Public Key (e) and (n)
 *   4. "Signature verified: true/false"
 *   5. The outgoing JSON response
 *   6. "Number of Blocks on Chain == N"
 *
 * Verification approach adapted from ShortMessageVerify.java (course materials).
 * RSAExample.java from course materials used as structural reference.
 * Code assistance provided by Claude AI (claude-sonnet-4-6).
 */

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.*;
import java.math.BigInteger;
import java.net.ServerSocket;
import java.net.Socket;
import java.security.MessageDigest;
import java.sql.Timestamp;
import java.util.Scanner;

public class VerifyingServerTCP {

    private static final int PORT = 7777;

    // Blockchain persists for the lifetime of the server process
    private static BlockChain blockchain = new BlockChain();

    private static final Gson gson = new Gson();

    public static void main(String[] args) {
        // Initialize with genesis block
        Block genesis = new Block(0, new Timestamp(System.currentTimeMillis()), "Genesis", 2);
        blockchain.addBlock(genesis);

        System.out.println("Blockchain server running on port " + PORT);

        try (ServerSocket listenSocket = new ServerSocket(PORT)) {
            while (true) {
                Socket clientSocket = listenSocket.accept();
                handleClient(clientSocket);
            }
        } catch (IOException e) {
            System.out.println("Server IO error: " + e.getMessage());
        }
    }

    /**
     * Handles all requests from a single client connection.
     * Reads one JSON request per line, verifies identity and signature,
     * processes the operation (if valid), and sends back a JSON response.
     *
     * @param clientSocket the connected client socket
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

                RequestMessage request = gson.fromJson(jsonRequest, RequestMessage.class);

                // Display the client's public key components
                String pubE = request.getPublicKeyE();
                String pubN = request.getPublicKeyN();
                System.out.println("Client Public Key (e): " + pubE);
                System.out.println("Client Public Key (n): " + pubN);

                ResponseMessage response;
                String operation = request.getOperation();

                // "exit" does not need signature verification
                if ("exit".equals(operation)) {
                    response = new ResponseMessage("success", "Server acknowledged exit");
                    String exitJson = gson.toJson(response);
                    System.out.println("Signature verified: N/A (exit)");
                    System.out.println(exitJson);
                    out.println(exitJson);
                    out.flush();
                    System.out.println("Number of Blocks on Chain == " + blockchain.getChainSize());
                    return;
                }

                // ---- Perform both security checks ----
                boolean idValid = verifyClientID(pubE, pubN, request.getClientID());
                boolean sigValid = false;
                if (idValid) {
                    sigValid = verifySignature(request);
                }

                System.out.println("Signature verified: " + (idValid && sigValid));

                if (!idValid || !sigValid) {
                    response = new ResponseMessage("error", "Error in request");
                } else {
                    // Both checks passed — process the operation
                    switch (operation) {
                        case "status":  response = handleStatus();         break;
                        case "add":     response = handleAdd(request);     break;
                        case "verify":  response = handleVerify();         break;
                        case "view":    response = handleView();           break;
                        case "corrupt": response = handleCorrupt(request); break;
                        case "repair":  response = handleRepair();         break;
                        default:
                            response = new ResponseMessage("error",
                                    "Unknown operation: " + operation);
                    }
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

    // ===================== Security Checks =====================

    /**
     * Verifies that the supplied clientID equals the last 20 bytes (hex) of
     * SHA-256(publicKeyE + publicKeyN).
     *
     * @param pubE     the claimed public key exponent e (decimal string)
     * @param pubN     the claimed public key modulus n (decimal string)
     * @param clientID the claimed client identity (40-char hex string)
     * @return true if the ID matches; false otherwise
     */
    private static boolean verifyClientID(String pubE, String pubN, String clientID) {
        if (pubE == null || pubN == null || clientID == null) return false;
        try {
            String enString = pubE + pubN;
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(enString.getBytes("UTF-8")); // 32 bytes

            // Take LAST 20 bytes and hex-encode them
            StringBuilder sb = new StringBuilder();
            for (int i = hash.length - 20; i < hash.length; i++) {
                sb.append(String.format("%02x", hash[i]));
            }
            return sb.toString().equals(clientID);
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Verifies the RSA signature on the request.
     *
     * Algorithm (adapted from ShortMessageVerify.java, course materials):
     *   1. Rebuild messageToSign = operation + index + difficulty + data.
     *   2. SHA-256 hash it; prepend 0x00 byte → expectedHashInt.
     *   3. RSA-decrypt signature with (e, n): decryptedHashInt = sig^e mod n.
     *   4. Return (expectedHashInt == decryptedHashInt).
     *
     * @param request the incoming request (must have publicKeyE, publicKeyN, signature)
     * @return true if the signature is valid; false otherwise
     */
    private static boolean verifySignature(RequestMessage request) {
        if (request.getPublicKeyE() == null
                || request.getPublicKeyN() == null
                || request.getSignature() == null) {
            return false;
        }
        try {
            BigInteger pubE = new BigInteger(request.getPublicKeyE());
            BigInteger pubN = new BigInteger(request.getPublicKeyN());
            BigInteger sig  = new BigInteger(request.getSignature());

            // Decrypt the signature using the public key
            BigInteger decryptedHashInt = sig.modPow(pubE, pubN);

            // Reconstruct what the hash should be
            String messageToSign = request.getMessageToSign();
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = md.digest(messageToSign.getBytes("UTF-8")); // 32 bytes

            // Prepend 0x00 to get a positive BigInteger (same as signing)
            byte[] paddedHash = new byte[hashBytes.length + 1];
            paddedHash[0] = 0;
            System.arraycopy(hashBytes, 0, paddedHash, 1, hashBytes.length);
            BigInteger expectedHashInt = new BigInteger(paddedHash);

            return expectedHashInt.compareTo(decryptedHashInt) == 0;
        } catch (Exception e) {
            return false;
        }
    }

    // ===================== Operation Handlers (same as Task 1) =====================

    private static ResponseMessage handleStatus() {
        blockchain.computeHashesPerSecond();
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
        return new ResponseMessage("success",
                "Total execution time to add this block was " + elapsed + " milliseconds",
                "", 0);
    }

    private static ResponseMessage handleVerify() {
        long start = System.currentTimeMillis();
        String result = blockchain.isChainValid();
        long elapsed = System.currentTimeMillis() - start;
        return new ResponseMessage("success",
                "Chain verification: " + result
                        + "\nTotal execution time required to verify the chain was "
                        + elapsed + " milliseconds",
                "", elapsed);
    }

    private static ResponseMessage handleView() {
        return new ResponseMessage("success", "Blockchain retrieved",
                blockchain.toString(), 0);
    }

    private static ResponseMessage handleCorrupt(RequestMessage request) {
        int blockId = request.getIndex();
        String newData = request.getData();
        blockchain.getBlocks().get(blockId).setData(newData);
        return new ResponseMessage("success",
                "Block " + blockId + " now holds " + newData);
    }

    private static ResponseMessage handleRepair() {
        long start = System.currentTimeMillis();
        blockchain.repairChain();
        long elapsed = System.currentTimeMillis() - start;
        return new ResponseMessage("success",
                "Total execution time required to repair the chain was " + elapsed + " milliseconds",
                "", elapsed);
    }
}
