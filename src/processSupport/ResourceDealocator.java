package processSupport;

import java.lang.ref.Cleaner;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * For cleaning up memory.
 *
 * The class that calls this method should implement autoclosable with:
 *
 * @Override public void close() throws Exception { cleanable.clean(); }
 */
public class ResourceDealocator<T> implements Runnable {

    /**
     * The cleaner for all the arrays.
     */
    public static final Cleaner cleaner = Cleaner.create(Executors.defaultThreadFactory());

    private final T[] needsClosure;
    private final Consumer<T> close;

    /**
     * Creates a cleanable object for the proffered array.
     *
     * @param <T> an object that needs to be cleaned.
     * @param obj An object containing a tool of Type T that needs to be cleaned 
     * when the object is no longer in use.
     * @param needsClosure The thing that needs to be closed.
     * @param closeOperation The close operation.
     * @return A Cleanable that will clean up data when the object containing
     * that data is no longer accessible.
     */
    public static <T> Cleaner.Cleanable register(Object obj, Consumer<T> closeOperation, T... needsClosure) {
        return cleaner.register(obj, new ResourceDealocator(closeOperation, needsClosure));
    }

    /**
     * Creates a memory cleaner.
     *
     * @param needsClosure The thing that needs to be cleaned from memory.
     * @param close The way in which it is to be cleaned.
     */
    private ResourceDealocator(Consumer<T> close, T... needsClosure) {
        this.needsClosure = needsClosure;
        this.close = close;
    }

    private final AtomicBoolean cleaned = new AtomicBoolean(false);

    @Override
    public void run() {
        if (needsClosure != null && cleaned.compareAndSet(false, true)) {
            for(T closeMe: needsClosure)
            close.accept(closeMe);
//            System.out.println("gpu.ResourceDealocator.run(): dealocated " + needsClosure.toString());
        }
    }

}
