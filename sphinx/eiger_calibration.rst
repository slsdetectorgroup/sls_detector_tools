EIGER Calibration
============================

This document describes the calibration procedure for an EIGER detector. Functions
are documented in the order they are normally run.

Calibration in the big X-ray box
-----------------------------------
 
When calibrating in the big xray box we can control the X-ray box directly form
the calibration scripts. Using the following functions automatically selects
the correct target and open/close the shutter. 
 
.. autofunction:: sls_detector_tools.calibration.do_vrf_scan

.. autofunction:: sls_detector_tools.calibration.do_scurve

.. autofunction:: sls_detector_tools.calibration.do_scurve_fit