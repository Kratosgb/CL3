import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.Scanner;

public class Client {
    public static void main(String[] args) {
        try {
            Registry registry = LocateRegistry.getRegistry("localhost", 1099);

            StringConcat stub = (StringConcat) registry.lookup("ConcatService");

            Scanner sc = new Scanner(System.in);

            System.out.print("Enter first string: ");
            String s1 = sc.nextLine();

            System.out.print("Enter second string: ");
            String s2 = sc.nextLine();

            String result = stub.concatenate(s1, s2);

            System.out.println("Concatenated String: " + result);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}