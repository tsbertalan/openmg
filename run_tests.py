"""Script runs tests and demos, checking quickly for regressions."""

from openmg.tests import doTests
doTests()
print

print "openmg_usage_demo:"
import openmg_usage_demo
openmg_usage_demo.simpleDemo()
print
openmg_usage_demo.main()
