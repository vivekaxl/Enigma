from models import *
from smote import *
from scipy.spatial.distance import euclidean
from random import choice
from numpy import sum
from sk import *

class Slots():
  "Place to read/write named slots."
  id = -1
  def __init__(i,**d) : 
    i.id = Slots.id = Slots.id + 1
    i.override(d)
  def override(i,d): i.__dict__.update(d); return i
  def __eq__(i,j)  : return i.id == j.id   
  def __ne__(i,j)  : return i.id != j.id   
  def __repr__(i)  : return '{' + showd(i.__dict__) + '}'


def xtile(lst,lo=0,hi=100,width=80,
             chops=[0.1 ,0.3,0.5,0.7,0.9],
             marks=["-" ," "," ","-"," "],
             bar="|",star="*",show=" %3.0f"):
  """The function _xtile_ takes a list of (possibly)
  unsorted numbers and presents them as a horizontal
  xtile chart (in ascii format). The default is a 
  contracted _quintile_ that shows the 
  10,30,50,70,90 breaks in the data (but this can be 
  changed- see the optional flags of the function).
  """
  def pos(p)   : return ordered[int(len(lst)*p)]
  def place(x) : 
    return int(width*float((x - lo))/(hi - lo+0.00001))
  def pretty(lst) : 
    return ', '.join([show % x for x in lst])
  ordered = sorted(lst)
  lo      = min(lo,ordered[0])
  hi      = max(hi,ordered[-1])
  what    = [pos(p)   for p in chops]
  where   = [place(n) for n in  what]
  out     = [" "] * width
  for one,two in pairs(where):
    for i in range(one,two): 
      out[i] = marks[0]
    marks = marks[1:]
  out[int(width/2)]    = bar
  out[place(pos(0.5))] = star 
  return '('+''.join(out) +  ")," +  pretty(what)

def _tileX() :
  import random
  random.seed(1)
  nums = [random.random()**2 for _ in range(100)]
  print xtile(nums,lo=0,hi=1.0,width=25,show=" %5.2f")


by   = lambda x: random.uniform(0,x)
def lo(m,x)      : return m.minR[x]
def hi(m,x)      : return  m.maxR[x]
def trim(m,x,i)  : # trim to legal range
    return max(lo(m,i), x%hi(m,i))

def gs(lst) : return [g(x) for x in lst]
def g(x)    : 
  if(x == None): return float(-1)
  return float('%g' % x) 

def extrapolate(m,one, two):
  def norm(m,energy):
  	return (energy-m.minVal)/(m.maxVal-m.minVal)
  count = 0
  tempdec = [0 for _ in xrange(m.n)]
  for a, b in zip(one.dec, two.dec):
  	tempdec[count] = trim(m,min(a, b) + random.random() * (abs(a - b)),count)
  	count += 1

  tempobj = [min(a, b) + random.random() * (abs(a - b)) for a, b in zip(one.obj, two.obj)]
  return Slots(changed = True,
            scores=norm(m,np.sum(tempobj)), 
            xblock=-1, #sam
            yblock=-1,  #sam
            x=-1,
            y=-1,#            
            obj = tempobj, #This needs to be removed. Not using it as of 11/10
            dec = tempdec)

def showd(d):
  "Pretty print a dictionary."
  def one(k,v):
    if isinstance(v,list):
      v = gs(v)
    if isinstance(v,float):
      return ":%s %g" % (k,v)
    return ":%s %s" % (k,v)
  return ' '.join([one(k,v) for k,v in
                    sorted(d.items())
                     if not "_" in k])

def some(m,x) :
  "with variable x of model m, pick one value at random" 
  return lo(m,x) + by(hi(m,x) - lo(m,x))

def candidate(m,d=[]):
  "Return an unscored individual."
  if len(d) == 0: d = [some(m,d) for d in xrange(m.n)]
  return Slots(changed = True,
            scores=1e6, 
            xblock=-1, #sam
            yblock=-1,  #sam
            x=-1,
            y=-1,
            obj = [None] * m.objf, #This needs to be removed. Not using it as of 11/10
            dec = d)

def score(m, individual): #model
  if individual.changed == False: 
    return individual.scores
  temp = m.evaluate(individual.dec) 
  #print "Score| score: ",temp[-1]
  return temp

def scores(m,t):
  "Score an individual."
  if t.changed:
    t.scores = score(m,t)
    t.changed = False
  return t.scores

def return_points(model,num_points):
  pop = []
  if num_points > 0:
    for _ in xrange(num_points):
      one = candidate(model)
      temp = scores(model,one)
      one.obj = temp[-model.objf-1:-1]
      one.scores = temp[-1]
      pop += [one]
  return pop


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#SMOTE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def populate(m,data,k = 3 ,reps = 1000):
	newData = []
	for _ in xrange(reps):
	  for one in data:
	    neigh = knn(one, data)[1:k + 1];
	    two = choice(neigh)
	    newData.append(extrapolate(m,one, two))
	data.extend(newData)
	return ([choice(data) for _ in xrange(1000)])

def _populate():
	model = DTLZ1()
	data = return_points(model,10)
	temp = populate(data)
	print len(temp)
	print temp[0]

def knn(one, two):
	pdistVect = []
	onearr = one.dec + one.obj
	#    set_trace()
	for ind, n in enumerate(two):
		narr = n.dec + n.obj
		pdistVect.append([ind, euclidean(onearr,narr)])
	indices = sorted(pdistVect, key = lambda F:F[1])
	return [two[n[0]] for n in indices]

def _knn():
	model = DTLZ1()
	one = return_points(model,1)
	two = return_points(model,10)
	print [x.dec for x in knn(one[0],two)]

def checkpoints(m,point):
	actual = m.evaluate(point.dec)[-1]
	predicted = point.scores
	return abs(actual-predicted)/actual

def checksmote():
	errorCollector = {}
	for klass in [DTLZ1,DTLZ5,DTLZ6,DTLZ7,Viennet,Osyczka,Fonseca,Kursawe,ZDT1,ZDT3]:
		print "Running ",klass.__name__
		model = klass()
		model.minVal,model.maxVal = model.baseline(model.minR, model.maxR)
		points = return_points(model,10)
		madeup = populate(model,points)
		error = [checkpoints(model,x) for x in madeup]
		errorCollector[klass.__name__]=error
	callrdivdemo(errorCollector)


def callrdivdemo(eraCollector,show="%5.2f"):
  #print eraCollector
  #print "callrdivdemo %d"%len(eraCollector.keys())
  keylist = eraCollector.keys() 
  #print keylist
  variant = len(keylist)
  #print variant
  rdivarray=[]
  for y in xrange(variant):
      #print "Length of array: %f"%len(eraCollector[keylist[y]][x])
      temp = eraCollector[keylist[y]]
      #print temp
      temp.insert(0,str(keylist[y]))
      #print temp
      rdivarray.append(temp)
  rdivDemo(rdivarray,show)




if __name__ == '__main__':
  random.seed(0)
  #_populate()
  checksmote()
  #_tileX()

"""
Output:

        rank ,          name ,            med   ,         iqr 
----------------------------------------------------
           1 ,         DTLZ5 ,          0  ,                0      (*                   |                   ), 0.00,  0.00,  0.00,  0.00,  0.00
           1 ,         DTLZ6 ,          0  ,                0      (*                   |                   ), 0.00,  0.00,  0.00,  0.00,  0.00
           1 ,         DTLZ1 ,          0  ,             0.02      (*                   |                   ), 0.00,  0.00,  0.00,  0.01,  0.09
           1 ,       Fonseca ,          0.01  ,          0.05      (*                   |                   ), 0.00,  0.00,  0.01,  0.03,  0.26
           1 ,       Osyczka ,          0.01  ,          0.05      (*                   |                   ), 0.00,  0.00,  0.01,  0.03,  0.34
           1 ,         DTLZ7 ,          0.07  ,          0.09      (*                   |                   ), 0.01,  0.04,  0.07,  0.11,  0.20
           1 ,          ZDT1 ,          0.08  ,          0.09      (*                   |                   ), 0.02,  0.05,  0.08,  0.12,  0.19
           1 ,          ZDT3 ,          0.11  ,          0.18      (*                   |                   ), 0.02,  0.06,  0.11,  0.20,  0.36
           2 ,       Kursawe ,          0.21  ,          0.21      (*                   |                   ), 0.05,  0.13,  0.21,  0.30,  0.56
           3 ,       Viennet ,          0.77  ,           0.4      (*                   |                   ), 0.21,  0.59,  0.77,  0.89,  1.88
"""  
