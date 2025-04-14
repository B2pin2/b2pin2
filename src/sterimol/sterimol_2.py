# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Comments and/or additions are welcome:
# kelvin.jackson@chem.ox.ac.uk or robert.paton@chem.ox.ac.uk

###############################################################
#                         sterimol.py                         #
#                                                             #
###############################################################
####  Written by: Dr Kelvin Jackson and Prof Robert Paton #####
####  Last modified:  Nov 09, 2016 ############################
###############################################################

import sys
sys.path.append('../')

from src.sterimol.sterimoltools_2 import calcSterimol, calcSandwich  # sterimoltoolsのインポート

def run_sterimol(file, radii="cpk", atom1=1, atom2=2, jobtype=1):
   """
   Notebook上でSterimolパラメータを計算する関数
   """
   if jobtype == 1:
      file_Params = calcSterimol(file, radii, atom1, atom2, True)
      L = file_Params.lval
      B1 = file_Params.B1
      B5 = file_Params.newB5
      print("\n   STERIMOL: using", radii, "van der Waals parameters")
      print("\n", "   Structure".ljust(25), "L".rjust(9), "B1".rjust(9), "B5".rjust(9))
      print("   " + file.ljust(22), "  %.2f".rjust(9) % L, "  %.2f".rjust(9) % B1, "  %.2f".rjust(9) % B5)
      print("")
      return "%.2f" % L, "%.2f" % B1, "%.2f" % B5 
    
   if jobtype == 2:
      print("\n   Sandwich Analysis\n   STERIMOL: using original CPK Van der Waals parameters\n")
      print("   " + "Structure".ljust(25), "Tolman_CA".rjust(9), "MC_dist".rjust(9), "L".rjust(9), "B1".rjust(9), "B5".rjust(9))
      for file in files:
         calcSandwich(file)
      print("\n")
   
