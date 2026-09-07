def accuracy(
        preds,
        labels
):

    correct = 0


    for p,y in zip(preds,labels):

        if p == y:
            correct += 1


    return correct / len(labels)

def precision(
        preds,
        labels,
        num_classes
):

    result=[]


    for cls in range(num_classes):

        TP=0
        FP=0


        for p,y in zip(preds,labels):

            if p==cls and y==cls:
                TP+=1

            elif p==cls and y!=cls:
                FP+=1


        if TP+FP==0:
            result.append(0)

        else:
            result.append(
                TP/(TP+FP)
            )


    return sum(result)/num_classes

def recall(
        preds,
        labels,
        num_classes
):

    result=[]


    for cls in range(num_classes):

        TP=0
        FN=0


        for p,y in zip(preds,labels):

            if p==cls and y==cls:
                TP+=1

            elif p!=cls and y==cls:
                FN+=1


        if TP+FN==0:
            result.append(0)

        else:
            result.append(
                TP/(TP+FN)
            )


    return sum(result)/num_classes

def f1(
        p,
        r
):

    if p+r==0:
        return 0

    return 2*p*r/(p+r)