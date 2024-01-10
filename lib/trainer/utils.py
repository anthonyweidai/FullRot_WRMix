
def getSetPath(TrainSet, TestSet, Split):
    # for source and target domain set path init
    if isinstance(TrainSet[0], str):
        TrainSet = TrainSet
        TestSet = TestSet
    else:
        TrainSet = TrainSet[Split]
        TestSet = TestSet[Split]
    
    return TrainSet, TestSet