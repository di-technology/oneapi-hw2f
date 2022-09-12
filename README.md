# oneapi-hw2f
This is oneAPI implementation of a 2-factor Hull-White model which will run on GPU and CPU. In Hull-White model the short rate $r$ is given by
$$r = x + y + \phi$$
where $x$ and $y$ follow the stochastic differential equations
$$dx = -\kappa_x dt + \sigma_x dW_x,$$
$$dy = -\kappa_y dt + \sigma_y dW_x,$$
$$dW_x dW_y = \rho dt.$$
In general $\kappa_x$, $\kappa_y$, $\sigma_x$, $\sigma_y$, $\rho$ can be time dependent. Very often $\kappa_x$, $\kappa_y$, $\rho$ are chosen as exogeneous parameters.
In this example for simplicity we choose all parameters as constant.

The easiest way to test the code is to build and run it on Intel DevCloud.

Once logged into DevCloud start the interactive job:
```
qsub -I -l nodes=1:gpu:gen9:ppn=2 -d .
```

Modify the source code by choosing either `cpu_selector` or `gpu_selector` on line 103 and 104. Run the `make` command. It will generate the executable `hw2f-usm`.
Run:
```
./hw2f-usm
```
It will output for example for GPU:
```
Running on device: Intel(R) UHD Graphics P630 [0x3e96]
Number of paths: 10000
PV: 0.419257
elapsed time: 0.239575s
The Monte Carlo simulation is successfully completed on the device.
```
Running the code for CPU will output:
```
Running on device: Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz
Number of paths: 10000
PV: 0.419257
elapsed time: 0.117267s
The Monte Carlo simulation is successfully completed on the device.
```
By default the program runs 10,000 Monte Carlo paths and generates 40 time steps. This is equivalent to 10 years trade with quarterly payments.

As you can see we are running the same source code on two different architectures. On different devices the performance of the code will be different.
