from bridge import Engine

def fixture():
 e=Engine.fixture()
 e.galleries['gal:north']['slots']=[f'N{i}' for i in range(1,7)]
 e.galleries['gal:south']['slots']=[f'S{i}' for i in range(1,7)]
 e.galleries['gal:archive']['slots']=[f'A{i}' for i in range(1,7)]
 e.rules['rule:research']={'version':1,'latest_due':'2027-03-31'}
 e.galleries['gal:archive']['rules']=['rule:root','rule:research','rule:archive']
 extras=[('banner','Silk Banner','textile'),('herons','Bronze Herons','metal'),('clockwork','Bronze Clockwork','metal'),('atlas-stars','Celestial Atlas of Stars','paper'),('orchids','Orchid Embroideries','textile'),('jade-copy','Jade Tablets','stone'),('map','River Map','paper'),('map-old','Old River Map','paper'),('bowl','Silver Bowl','metal'),('bowl-blue','Silver Bowl Blue','metal'),('vase','Stone Vase','stone'),('vases','Stone Vases','stone')]
 for key,title,medium in extras:
  e.sequence+=1;lid='loan:'+key
  e.loans[lid]={'id':lid,'target_id':'seed:'+key,'title':title,'home':'gal:archive','medium':medium,'restricted':False,'reservation':None,'notes':[],'creation':{'sequence':e.sequence,'event_id':'fixture','actor':'fixture'}}
 e.rebuild_indexes();return e
