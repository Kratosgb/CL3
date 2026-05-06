import java.rmi.Remote;
import java.rmi.RemoteException;

public interface StringConcat extends Remote {
    public String concatenate(String s1, String s2) throws RemoteException;
}