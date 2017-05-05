# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:59:48 2017

@author: echo
"""


# Python standard library imports

# Third-party imports

import requests
from pyrosutilities import getxmlval


from lxml import etree



def post_request(rname,strat,priority,pwd):
    '''Builds and posts a request on CADOR for the user alert. This just creates the request
    for the new scenes to be posted to. Returns the idreq of the new request'''
    strat = "0"
    priority = "0"
    depotstring = build_request(rname,strat,priority,pwd)
    files = {'file': ('marequete.xml', depotstring)}
    ans = requests.post("http://cador.obs-hp.fr/ros/manage/rest/cador.php/", files=files)
    depot = etree.XML(ans.content)
    idreq = getxmlval(depot,"idreq")[0]
    print("Successfuly added request:",idreq)
    return idreq
    
def post_scene(prefix,idreq,entry,exps,filters,pwd):
    '''This turns the scene table line entry into a scene for CADOR
    and posts it to the server'''
    ddate = "1"
    processing = "0"
    spriority = "0"
    depotstring, ra, dec, timeisot = build_scene(prefix,idreq,entry,exps,filters,spriority,processing,ddate,pwd)
    files = {'file': ('mascene.xml', depotstring)}
    ans = requests.post("http://cador.obs-hp.fr/ros/manage/rest/cador.php/"+str(idreq), files=files)
    #print (ans.content)
    depot = etree.XML(ans.content)
    try :
        idscene = getxmlval(depot,"idscene")[0]
    except:
        print("fail to parse response in xml")
        return
    print("Successfuly added scene:",idscene)

    return idscene, ra, dec, timeisot
def build_scene(prefix,idreq,entry,exps,filters,spriority,processing,ddate,pwd):
    '''Creates the XML string to submit a new scene to the CADOR REST API'''
    #print()
    ra = str((int)(entry["coords"].ra.hms[0])) + ":" +\
        str((int)(entry["coords"].ra.hms[1])) + ":" +\
        str((int)(entry["coords"].ra.hms[2]))
    dec = str((int)(entry["coords"].dec.dms[0])) + ":" +\
        str((int)(abs(entry["coords"].dec.dms[1]))) + ":" +\
        str((int)(abs(entry["coords"].dec.dms[2])))
    timeisot = entry["time"].isot
    #print(ra,dec,timeisot)
    parameters = {
    "description":"Depot de scene pour CADOR",
    "versionmsg":"",
    "login":"alert",
    "passwd":pwd,  
    "type":"AL",  
    "sname": prefix + str(entry["idtelescope"]) + "_" + str(entry["index"]) + "_" + str(entry["tindex"]) + "_",  
    "ra":ra,  
    "decl":dec,  
    "altmin":"10",  
    "moonmin":"10",  
    "t1":exps[0],  
    "t2":exps[1],  
    "t3":exps[2],  
    "t4":exps[3],  
    "t5":exps[4],  
    "t6":exps[5],  
    "f1":filters[0],  
    "f2":filters[1],  
    "f3":filters[2],  
    "f4":filters[3],  
    "f5":filters[4],  
    "f6":filters[5],  
    "dra1":"0.00418098", 
    "dra2":"0.00418098",  
    "dra3":"0.00418098",  
    "dra4":"0.00418098",  
    "dra5":"0.00418098",  
    "dra6":"0.00418098", 
    "ddec1":"0",
    "ddec2":"0",
    "ddec3":"0",
    "ddec4":"0",
    "ddec5":"0",
    "ddec6":"0",
    "spriority":"0",  
    "idtelescope":entry["idtelescope"], 
    "processing":processing,
    "date":timeisot,
    "ddate":ddate }
    #Creating an xml root
    depot = etree.Element("depotcador")
    #Inserting the xml elements
    for element in parameters.keys():
        etree.SubElement(depot, element).text = str(parameters[element])
    #Converting to a string
    stringform = etree.tostring(depot,pretty_print= True,xml_declaration=True,encoding="UTF-8")
    #print(stringform)
    return stringform, ra ,dec, timeisot
    
def build_request(rname,strat,priority,pwd):
    '''Creates the XML REST string for a new request '''
    parameters = {"description":"Depot de requete pour CADOR",
                  "versionmsg": "req0.1",
                  "rname":      "rname",
                  "login":      "alert",
                  "passwd":     "monpassword",
                  "strategy":   "0",
                  "rpriority":  "0"}
    parameters["passwd"] = pwd
    parameters["rname"] = rname
    parameters["strategy"] = strat
    parameters["priority"] = priority
    depot = etree.Element("depotcador")
    parameters
    for element in parameters.keys():
        etree.SubElement(depot, element).text = parameters[element]

    
    print(etree.tostring(depot,pretty_print= True,xml_declaration=True,encoding="UTF-8"))
    stringform = etree.tostring(depot,pretty_print= True,xml_declaration=True,encoding="UTF-8")
    return stringform
    
def remove_request(idreq,pwd):
    '''This will remove the request from CADOR and delete all the scenes'''
    parameters = {"description":"Depot de requete pour CADOR",
              "versionmsg": "req0.1",
              "rname":      "rname",
              "login":      "alert",
              "passwd":     pwd
              }
    depot = etree.Element("depotcador")
    for element in parameters.keys():
        etree.SubElement(depot, element).text = parameters[element]
    stringform = etree.tostring(depot,pretty_print= True,xml_declaration=True,encoding="UTF-8")
    
    files = {'file': ('identity_v01.xml', stringform)}
    ans = requests.post("http://cador.obs-hp.fr/ros/manage/rest/cador.php/" + str(idreq) + "/del", files=files)
    print (ans.content)
    depot = etree.XML(ans.content)
    deleted_requests = getxmlval(depot,"deleted_requests")[0]
    deleted_scenes = getxmlval(depot,"deleted_scenes")[0]
    print("Successfuly deleted requests and scenes:",deleted_requests,deleted_scenes)
    return idreq
