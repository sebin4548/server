def average(detection_results3):
    listX1,listX2, listY1, listY2 = [],[],[],[]
    listaX1, listaX2, listaY1, listaY2 = [],[],[],[]
    for result in detection_results3:
        num_Runner = sum(1 for obj in result.boxes if obj.cls == 0.)
        for obj in (result.boxes):
            if num_Runner == 1 and obj.cls == 0.:
                x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                listX1.append(x1)
                listX2.append(x2)
                listY1.append(y1)
                listY2.append(y2)
            if num_Runner == 1 and obj.cls == 3.:
                x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
                listaX1.append(x1)
                listaX2.append(x2)
                listaY1.append(y1)
                listaY2.append(y2)

    R0x1, R0y1, R0x2, R0y2 = sum(listX1)/len(listX1), sum(listY1)/len(listY1), sum(listX2)/len(listX2), sum(listY2)/len(listY2)
    R3x1, R3y1, R3x2, R3y2 = sum(listaX1)/len(listaX1), sum(listaY1)/len(listaY1), sum(listaX2)/len(listaX2), sum(listaY2)/len(listaY2)
    return R0x1, R0y1, R0x2, R0y2, R3x1, R3y1, R3x2, R3y2
