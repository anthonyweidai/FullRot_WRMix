
def checkExp(ModelConfigDict: dict):
    CfgDict1 = ModelConfigDict[list(ModelConfigDict)[-1]]
    CfgDict2 = ModelConfigDict[list(ModelConfigDict)[-2]]
    if CfgDict1['in'] != CfgDict1['out'] and \
        CfgDict1['stage'] == CfgDict2['stage']:
            return True
    else:
        return False


def getLastIdxFromStage(ModelConfigDict: dict, Stage: int): # , CheckExpMode: bool=False
        # get the last layer Index among those layers with the same stride stages
        StageOri = Stage
        if StageOri < 0:
            # MaxStage = ModelConfigDict[list(ModelConfigDict)[-1]]['stage']
            Stage = ModelConfigDict[list(ModelConfigDict)[-1]]['stage'] + Stage + 1
            # if CheckExpMode:
            #     if checkExp(ModelConfigDict) and StageOri != -1:
            #         Stage -= 1
                
        StageIdx = -1
        for Config in ModelConfigDict:
            if ModelConfigDict[Config]['stage'] <= Stage:
                StageIdx += 1
            else:
                break
            
        return StageIdx
    

def getChannelsbyStage(ModelConfigDict: dict, StageIdx: int,):
    return ModelConfigDict[list(ModelConfigDict)[StageIdx]]['out']


def callDictbyStage(ModelConfigDict: dict, Stage: int):
    StageIdx = getLastIdxFromStage(ModelConfigDict, Stage)
    return getChannelsbyStage(ModelConfigDict, StageIdx)
