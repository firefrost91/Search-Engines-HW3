/**
 * SigningClientTCP.java
 *
 * Author: Student Name
 * Email: student@andrew.cmu.edu
 *
 * Task 2 blockchain client with RSA digital signatures.
 *
 * On startup, this client:
 *   1. Generates a fresh RSA key pair (2048-bit primes for security).
 *   2. Displays e, d, and n to the user.
 *   3. Computes and displays the clientID =
 *        last 20 bytes (hex) of SHA-256(e.toString() + n.toString()).
 *
 * For every request, the client:
 *   a. Builds a RequestMessage with operation, index, difficulty, data.
 *   b. Sets clientID, publicKeyE, publicKeyN.
 *   c. Computes messageToSign = operation + index + difficulty + data.
 *   d. SHA-256 hashes messageToSign, prepends a 0x00 byte so the result
 *      is a positive BigInteger, then RSA-encrypts with (d, n) to create
 *      the signature.
 *   e. Sets signature and sends the fully populated RequestMessage.
 *
 * The server verifies ID and signature before servicing any request.
 *
 * RSA algorithm reference: Cormen, Leiserson, Rivest, Stein (CLR).
 * RSAExample.java from course materials used as structural reference.
 * ShortMessageSign.java signing approach adapted for full SHA-256 hashes.
 * Code assistance provided by Claude AI (claude-sonnet-4-6).
 */

import com.google.gson.Gson;

import java.io.*;
import java.math.BigInteger;
import java.net.Socket;
import java.security.MessageDigest;
import java.util.Random;
import java.util.Scanner;

public class SigningClientTCP {

    private static final int SERVER_PORT = 7777;
    private static final Gson gson = new Gson();

    // RSA key components — generated fresh each run
    private static BigInteger e;   // public exponent
    private static BigInteger d;   // private exponent
    private static BigInteger n;   // modulus (public and private)

    // Client identity derived from public key
    private static String clientID;

    public static void main(String[] args) {
        String host = (args.length > 0) ? args[0] : "localhost";

        // Step 1 — Generate RSA key pair
        generateRSAKeys();

        // Step 2 — Compute clientID = last 20 bytes of SHA-256(e || n)
        clientID = computeClientID();

        // Step 3 — Display keys and ID to the user
        System.out.println("Client Key Pair:");
        System.out.println("Public Key (e): " + e);
        System.out.println("Public Key (n): " + n);
        System.out.println("Private Key (d): " + d);
        System.out.println("Client ID: " + clientID);

        // Step 4 — Connect and interact
        try (
            Socket socket = new Socket(host, SERVER_PORT);
            Scanner networkIn = new Scanner(socket.getInputStream());
            PrintWriter networkOut = new PrintWriter(
                new BufferedWriter(new OutputStreamWriter(socket.getOutputStream())));
            Scanner keyboard = new Scanner(System.in)
        ) {
            int choice = -1;
            while (choice != 6) {
                System.out.println("0. View basic blockchain status.");
                System.out.println("1. Add a transaction to the blockchain.");
                System.out.println("2. Verify the blockchain.");
                System.out.println("3. View the blockchain.");
                System.out.println("4. Corrupt the chain.");
                System.out.println("5. Hide the corruption by repairing the chain.");
                System.out.println("6. Exit");

                choice = Integer.parseInt(keyboard.nextLine().trim());

                switch (choice) {
                    case 0:
                        sendAndDisplay(buildRequest("status"), networkOut, networkIn);
                        break;

                    case 1: {
                        System.out.println("Enter difficulty > 1");
                        String diff = keyboard.nextLine().trim();
                        System.out.println("Enter transaction");
                        String data = keyboard.nextLine();
                        sendAndDisplay(buildRequest("add", diff, data), networkOut, networkIn);
                        break;
                    }

                    case 2:
                        sendAndDisplay(buildRequest("verify"), networkOut, networkIn);
                        break;

                    case 3:
                        System.out.println("View the Blockchain");
                        sendViewAndDisplay(networkOut, networkIn);
                        break;

                    case 4: {
                        System.out.println("Corrupt the Blockchain");
                        System.out.println("Enter block ID of block to corrupt");
                        int blockId = Integer.parseInt(keyboard.nextLine().trim());
                        System.out.println("Enter new data for block " + blockId);
                        String newData = keyboard.nextLine();
                        RequestMessage req = buildRequest("corrupt", blockId, newData);
                        ResponseMessage resp = sendRequest(req, networkOut, networkIn);
                        if (resp != null) System.out.println(resp.getMessage());
                        break;
                    }

                    case 5:
                        sendAndDisplay(buildRequest("repair"), networkOut, networkIn);
                        break;

                    case 6:
                        sendRequest(buildRequest("exit"), networkOut, networkIn);
                        break;

                    default:
                        System.out.println("Invalid option. Please choose 0-6.");
                }
            }

        } catch (IOException ex) {
            System.out.println("Client IO error: " + ex.getMessage());
        }
    }

    // ===================== RSA Key Generation =====================

    /**
     * Generates a 2048-bit RSA key pair (e, d, n) using random primes.
     * Algorithm from CLR:
     *   1. Select large primes p and q.
     *   2. n = p * q
     *   3. phi = (p-1)*(q-1)
     *   4. e = 65537 (standard public exponent, odd and coprime to phi)
     *   5. d = e^{-1} mod phi
     *
     * RSAExample.java from course materials used as a reference.
     * (Claude AI assistance acknowledged.)
     */
    private static void generateRSAKeys() {
        Random rnd = new Random();
        // Use 1024-bit primes so that n is ~2048 bits, suitable for signing SHA-256 hashes
        BigInteger p = new BigInteger(1024, 100, rnd);
        BigInteger q = new BigInteger(1024, 100, rnd);

        n = p.multiply(q);
        BigInteger phi = (p.subtract(BigInteger.ONE)).multiply(q.subtract(BigInteger.ONE));

        e = new BigInteger("65537"); // Standard public exponent
        d = e.modInverse(phi);
    }

    // ===================== Client ID Computation =====================

    /**
     * Computes the clientID as the last 20 bytes (hex-encoded) of
     * SHA-256(e.toString() + n.toString()).
     * Modeled after Ethereum's address derivation from a public key.
     *
     * @return 40-character lowercase hex string (20 bytes)
     */
    private static String computeClientID() {
        try {
            String enString = e.toString() + n.toString();
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(enString.getBytes("UTF-8")); // 32 bytes

            // Take the LAST 20 bytes
            StringBuilder sb = new StringBuilder();
            for (int i = hash.length - 20; i < hash.length; i++) {
                sb.append(String.format("%02x", hash[i]));
            }
            return sb.toString();
        } catch (Exception ex) {
            throw new RuntimeException("clientID computation failed: " + ex.getMessage());
        }
    }

    // ===================== Request Building =====================

    /**
     * Builds and signs a RequestMessage for a no-parameter operation.
     *
     * @param operation the operation string
     * @return fully signed RequestMessage
     */
    private static RequestMessage buildRequest(String operation) {
        RequestMessage req = new RequestMessage(operation);
        return signRequest(req);
    }

    /**
     * Builds and signs a RequestMessage for the "add" operation.
     *
     * @param operation  "add"
     * @param difficulty the mining difficulty
     * @param data       the transaction string
     * @return fully signed RequestMessage
     */
    private static RequestMessage buildRequest(String operation, String difficulty, String data) {
        RequestMessage req = new RequestMessage(operation, difficulty, data);
        return signRequest(req);
    }

    /**
     * Builds and signs a RequestMessage for the "corrupt" operation.
     *
     * @param operation "corrupt"
     * @param index     the block index to corrupt
     * @param data      the new data
     * @return fully signed RequestMessage
     */
    private static RequestMessage buildRequest(String operation, int index, String data) {
        RequestMessage req = new RequestMessage(operation, index, data);
        return signRequest(req);
    }

    /**
     * Signs a RequestMessage by:
     *   1. Computing SHA-256 of getMessageToSign().
     *   2. Prepending a 0x00 byte to ensure a positive BigInteger.
     *   3. RSA-encrypting with (d, n) to produce the signature.
     * Also sets clientID, publicKeyE, publicKeyN.
     *
     * Signing approach adapted from ShortMessageSign.java (course materials).
     * (Claude AI assistance acknowledged.)
     *
     * @param req the request to sign (modified in-place)
     * @return the same request with signature fields populated
     */
    private static RequestMessage signRequest(RequestMessage req) {
        try {
            req.setClientID(clientID);
            req.setPublicKeyE(e.toString());
            req.setPublicKeyN(n.toString());

            // What gets signed: operation + index + difficulty + data
            String messageToSign = req.getMessageToSign();

            // Hash with SHA-256
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = md.digest(messageToSign.getBytes("UTF-8")); // 32 bytes

            // Prepend 0x00 to ensure the BigInteger is positive
            byte[] paddedHash = new byte[hashBytes.length + 1];
            paddedHash[0] = 0; // most-significant byte = 0 → positive
            System.arraycopy(hashBytes, 0, paddedHash, 1, hashBytes.length);

            BigInteger hashInt = new BigInteger(paddedHash);

            // Encrypt hash with private key (d, n) — this IS the signature
            BigInteger sig = hashInt.modPow(d, n);
            req.setSignature(sig.toString());

        } catch (Exception ex) {
            throw new RuntimeException("Signing failed: " + ex.getMessage());
        }
        return req;
    }

    // ===================== Network Helpers =====================

    private static void sendAndDisplay(RequestMessage request,
                                       PrintWriter out, Scanner in) {
        ResponseMessage response = sendRequest(request, out, in);
        if (response != null) System.out.println(response.getMessage());
    }

    private static void sendViewAndDisplay(PrintWriter out, Scanner in) {
        RequestMessage req = buildRequest("view");
        ResponseMessage response = sendRequest(req, out, in);
        if (response != null && response.getData() != null && !response.getData().isEmpty()) {
            System.out.println(response.getData());
        }
    }

    private static ResponseMessage sendRequest(RequestMessage request,
                                               PrintWriter out, Scanner in) {
        try {
            String json = gson.toJson(request);
            out.println(json);
            out.flush();
            if (in.hasNextLine()) {
                String responseJson = in.nextLine();
                return gson.fromJson(responseJson, ResponseMessage.class);
            }
        } catch (Exception ex) {
            System.out.println("Communication error: " + ex.getMessage());
        }
        return null;
    }
}
