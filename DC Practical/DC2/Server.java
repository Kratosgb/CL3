import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class Server {
    public static void main(String[] args) {
        try {
            StringConcatImpl obj = new StringConcatImpl();

            Registry registry = LocateRegistry.createRegistry(1099);
            registry.rebind("ConcatService", obj);

            System.out.println("Server is running...");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}