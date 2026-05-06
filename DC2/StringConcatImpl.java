import java.rmi.server.UnicastRemoteObject;
import java.rmi.RemoteException;

public class StringConcatImpl extends UnicastRemoteObject implements StringConcat {

    protected StringConcatImpl() throws RemoteException {
        super();
    }

    public String concatenate(String s1, String s2) throws RemoteException {
        return s1 + s2;
    }
}