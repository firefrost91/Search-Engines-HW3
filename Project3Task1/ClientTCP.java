/**
 * ClientTCP.java
 *
 * Author: Student Name
 * Email: student@andrew.cmu.edu
 *
 * Task 1 blockchain client. Presents the same menu-driven console interface
 * as the Task 0 standalone blockchain, but all operations are forwarded to
 * a remote server over a TCP socket using JSON (Gson) for serialization.
 *
 * The client implements a proxy design: the menu logic and display are
 * entirely client-side, but the blockchain lives on the server. The client
 * sends a RequestMessage JSON string and receives a ResponseMessage JSON string
 * for each menu choice.
 *
 * Usage: java ClientTCP [serverHost]
 *   Default host: localhost
 *
 * Note: Code assistance provided by Claude AI (claude-sonnet-4-6).
 */

import com.google.gson.Gson;

import java.io.*;
import java.net.Socket;
import java.util.Scanner;

public class ClientTCP {

    // Server connection details
    private static final int SERVER_PORT = 7777;

    // Gson for JSON serialization / deserialization
    private static final Gson gson = new Gson();

    public static void main(String[] args) {
        String host = (args.length > 0) ? args[0] : "localhost";

        try (
            Socket socket = new Socket(host, SERVER_PORT);
            Scanner in = new Scanner(socket.getInputStream());
            PrintWriter out = new PrintWriter(
                new BufferedWriter(new OutputStreamWriter(socket.getOutputStream())));
            Scanner keyboard = new Scanner(System.in)
        ) {
            int choice = -1;
            while (choice != 6) {
                // Display menu identical to Task 0
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
                        sendAndDisplay(new RequestMessage("status"), out, in);
                        break;

                    case 1: {
                        System.out.println("Enter difficulty > 1");
                        String difficulty = keyboard.nextLine().trim();
                        System.out.println("Enter transaction");
                        String data = keyboard.nextLine();
                        RequestMessage req = new RequestMessage("add", difficulty, data);
                        sendAndDisplay(req, out, in);
                        break;
                    }

                    case 2:
                        sendAndDisplay(new RequestMessage("verify"), out, in);
                        break;

                    case 3:
                        // "view" returns blockchain JSON in data field; display it
                        System.out.println("View the Blockchain");
                        sendViewAndDisplay(out, in);
                        break;

                    case 4: {
                        System.out.println("Corrupt the Blockchain");
                        System.out.println("Enter block ID of block to corrupt");
                        int blockId = Integer.parseInt(keyboard.nextLine().trim());
                        System.out.println("Enter new data for block " + blockId);
                        String newData = keyboard.nextLine();
                        RequestMessage req = new RequestMessage("corrupt", blockId, newData);
                        ResponseMessage resp = sendRequest(req, out, in);
                        System.out.println(resp.getMessage());
                        break;
                    }

                    case 5:
                        sendAndDisplay(new RequestMessage("repair"), out, in);
                        break;

                    case 6:
                        // Inform the server then exit
                        sendRequest(new RequestMessage("exit"), out, in);
                        break;

                    default:
                        System.out.println("Invalid option. Please choose 0-6.");
                }
            }

        } catch (IOException e) {
            System.out.println("Client IO error: " + e.getMessage());
        }
    }

    // ===================== Helper Methods =====================

    /**
     * Serializes a request, sends it to the server, reads the response,
     * and prints the response message to the console.
     * Used for operations whose display is simply the message field.
     *
     * @param request the RequestMessage to send
     * @param out     the output writer connected to the server
     * @param in      the input scanner connected to the server
     */
    private static void sendAndDisplay(RequestMessage request, PrintWriter out, Scanner in) {
        ResponseMessage response = sendRequest(request, out, in);
        if (response != null) {
            System.out.println(response.getMessage());
        }
    }

    /**
     * Sends the "view" request and prints the blockchain JSON from
     * the response's data field (not the message field).
     *
     * @param out the output writer connected to the server
     * @param in  the input scanner connected to the server
     */
    private static void sendViewAndDisplay(PrintWriter out, Scanner in) {
        ResponseMessage response = sendRequest(new RequestMessage("view"), out, in);
        if (response != null && response.getData() != null && !response.getData().isEmpty()) {
            System.out.println(response.getData());
        }
    }

    /**
     * Core method: serializes a RequestMessage to JSON, sends it over the
     * socket, reads the single-line JSON response, and deserializes it.
     *
     * @param request the request to send
     * @param out     the output writer connected to the server
     * @param in      the input scanner connected to the server
     * @return the deserialized ResponseMessage, or null on error
     */
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
        } catch (Exception e) {
            System.out.println("Communication error: " + e.getMessage());
        }
        return null;
    }
}
