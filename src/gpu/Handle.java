package gpu;

import java.lang.ref.Cleaner;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;

public class Handle implements AutoCloseable {

    private static final Cleaner cleaner = Cleaner.create();
    private final Cleaner.Cleanable cleanable;
    private cublasHandle handle;

    public Handle() {
        
        handle = new cublasHandle();
        JCublas2.cublasCreate(handle);        
        cleanable = ResourceDealocator.register(this, handle, handle -> JCublas2.cublasDestroy(handle));
    }

    public cublasHandle get() {
        return handle;
    }

    @Override
    public void close() throws Exception {
        cleanable.clean();
    }

}
