



### This script is a code example of how to request and download data from the Canadian
  # data portal API OceanNetworks 3.0 (https://wiki.oceannetworks.ca/display/O2A/Request+Data+Product) 
#### =================================================================================



# This example creates a request for a Audio Data data product for the 
# Barkley Canyon / Axis (POD 1) 150 kHz ADCP and returns information about the request. 
# The returns a Request Id, which can be used to run the data product.


import requests
import json
  
url = 'https://data.oceannetworks.ca/api/dataProductDelivery'
parameters = {'method':'request',
            'token':'9f536fe9-d2e8-4508-a92e-48c6d931b97f',# replace YOUR_TOKEN_HERE with 
                #your personal token obtained from the 'Web Services API' tab at https://data.oceannetworks.ca/Profile when logged in.
            'locationCode':'SGE.H1',             # Strait of Georgia / Channel 1 (H1)
            'deviceCategoryCode':'HYDROPHONE',    # ISO3 Hydrophone Array 01
            'dataProductCode':'AD',                 # Audio Data
            'dpo_hydrophoneChannel':'H1',           # Channel 1 (H1)
            'dpo_audioDownsample':16000,            # downsample data to 16000 Herz (Hz)
            'dpo_audioFormatConversion':1,
            'dpo_hydrophoneDataDiversionMode':'OD',
            'extension':'wav',                  # wav file
            'dateFrom':'2014-09-01T00:00:00.000Z',  # The datetime of the first data point (From Date)
            'dateTo':'2015-09-01T00:00:00.000Z'}    # The datetime of the last data point (To Date)
            #'dpo_qualityControl':1,             # The Quality Control data product option - See https://wiki.oceannetworks.ca/display/DP/1
            #'dpo_resample':'none',              # The Resampling data product option - See https://wiki.oceannetworks.ca/display/DP/1
            #'dpo_dataGaps':0}                   # The Data Gaps data product option - See https://wiki.oceannetworks.ca/display/DP/1

response = requests.get(url,params=parameters)
  
if (response.ok):
    requestInfo = json.loads(str(response.content,'utf-8')) # convert the json response to an object
     
    print('Request Id: {}'.format(requestInfo['dpRequestId']))      # Print the Request Id
     
    if ('numFiles' in requestInfo.keys()):
        print('File Count: {}'.format(requestInfo['numFiles']))     # Print the Estimated File Size
  
    if ('fileSize' in requestInfo.keys()):
        print('File Size: {}'.format(requestInfo['fileSize']))      # Print the Estimated File Size
     
    if 'downloadTimes' in requestInfo.keys():
        print('Estimated download time:')
        for e in sorted(requestInfo['downloadTimes'].items(),key=lambda t: t[1]):
            print('  {} - {} sec'.format(e[0],'{:0.2f}'.format(e[1])))
 
 
    if 'estimatedFileSize' in requestInfo.keys():
        print('Estimated File Size: {}'.format(requestInfo['estimatedFileSize']))
                 
    if 'estimatedProcessingTime' in requestInfo.keys():
        print('Estimated Processing Time: {}'.format(requestInfo['estimatedProcessingTime']))
  
else:
    if(response.status_code == 400):
        error = json.loads(str(response.content,'utf-8'))
        print(error) # json response contains a list of errors, with an errorMessage and parameter
    else:
        print ('Error {} - {}'.format(response.status_code,response.reason))


### next section (request the data):
# =====================

# results in:
'dpRequestId': 16862264

## now request the data
url = 'https://data.oceannetworks.ca/api/dataProductDelivery'      
parameters = {'method':'run',
            'token':'9f536fe9-d2e8-4508-a92e-48c6d931b97f',              # replace YOUR_TOKEN_HERE with your personal token obtained from the 'Web Services API' tab at https://data.oceannetworks.ca/Profile when logged in.
           'dpRequestId':16862264}     # replace YOUR_REQUEST_ID_HERE with a requestId number returned from the request method
response = requests.get(url,params=parameters)
  
if (response.ok):
    r = json.loads(str(response.content,'utf-8')) # convert the json response to an object
  
    for runId in [run['dpRunId'] for run in r]:
        print('Run Id: {}'.format(runId))       # Print each of the Run Ids
 
else:
    if(response.status_code == 400):
        error = json.loads(str(response.content,'utf-8'))
        print(error) # json response contains a list of errors, with an errorMessage and parameter
    else:
        print ('Error {} - {}'.format(response.status_code,response.reason))



### next section (download the data)
# =====================

# results in:
'dpRunId':36642238

## now download the data
import requests
import json
import os
import sys
from contextlib import closing
import errno
 
url = 'https://data.oceannetworks.ca/api/dataProductDelivery'
parameters = {'method':'download',
            'token':'9f536fe9-d2e8-4508-a92e-48c6d931b97f',   # replace YOUR_TOKEN_HERE with your personal token obtained from the 'Web Services API' tab at https://data.oceannetworks.ca/Profile when logged in..
            'dpRunId':36642238,       # replace YOUR_RUN_ID with the dpRunId returned from the 'run' method.
            'index':1}                   # for run requests that contain more than one file, change the index number to the index of the file you would like to download.
                                           # If the index number does not exist an HTTP 410 and a message will be returned.
 
 
outPath='d:/'                        # replace with the file location you would like the file to be downloaded to.
 
with closing(requests.get(url,params=parameters,stream=True)) as streamResponse:
    if streamResponse.status_code == 200: #OK
        if 'Content-Disposition' in streamResponse.headers.keys():
            content = streamResponse.headers['Content-Disposition']
            filename = content.split('filename=')[1]
        else:
            print('Error: Invalid Header')
            streamResponse.close()
            sys.exit(-1)
         
        filePath = '{}/{}'.format(outPath,filename)
        try:
            if (not os.path.isfile(filePath)):
                #Create the directory structure if it doesn't already exist
                try:
                    os.makedirs(outPath)
                except OSError as exc:
                    if exc.errno == errno.EEXIST and os.path.isdir(outPath):
                        pass
                    else:
                        raise
                print ("Downloading '{}'".format(filename))
 
                with open(filePath,'wb') as handle:
                    try:
                        for block in streamResponse.iter_content(1024):
                            handle.write(block)
                    except KeyboardInterrupt:
                        print('Process interupted: Deleting {}'.format(filePath))
                        handle.close()
                        streamResponse.close()
                        os.remove(filePath)
                        sys.exit(-1)
            else:
                print ("  Skipping '{}': File Already Exists".format(filename))
        except:
            msg = 'Error streaming response.'
            print(msg)
    else:
        if(streamResponse.status_code in [202,204,400,404,410]):
            payload = json.loads(str(streamResponse.content,'utf-8'))
            if len(payload) >= 1:
                msg = payload['message']
                print('HTTP {} - {}: {}'.format(streamResponse.status_code,streamResponse.reason,msg))
        else:
            print ('Error {} - {}'.format(streamResponse.status_code,streamResponse.reason))
 
streamResponse.close()