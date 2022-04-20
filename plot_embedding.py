import matplotlib.pyplot as plt
embeddings=[]
embeddings.append([-10.661215,-1.240579,-3.3444042])
embeddings.append([-6.6341405,0.010744736,1.347795])
embeddings.append([-5.642037,-3.736256,3.568873])
embeddings.append([-2.0516467,-0.92864585,7.6137214])
embeddings.append([2.1141756,1.082346,5.9462113])
embeddings.append([3.854389,0.2507965,7.3754015])
embeddings.append([5.274848,-0.9349187,8.321509])
embeddings.append([8.573753,0.32672,6.7929997])
embeddings.append([10.297417,-0.6896142,5.340005])
embeddings.append([10.213425,-3.9296908,6.39862])
# embeddings.append([14.38648,-9.580958,-16.041727,-1.164166,12.3197365,3.672143,-3.1414335,-4.0125556,-6.67028,0.63735306,-60.31507,-26.34062,-53.79265,16.057978,2.2778113,3.4116797])
# embeddings.append([4.8981857,-2.1659336,3.6557002,-2.8433738,14.906114,5.482699,-2.4826305,5.823681,1.2792983,-2.8405387,-25.147417,-27.718727,-52.27161,-6.88948,-0.24129571,8.554256])
# embeddings.append([4.5458994,-2.4300458,1.1819551,0.26374474,10.243903,-1.2598388,0.9122858,6.1226835,-5.1163387,2.9107726,-13.963078,-36.49345,-43.173885,0.9039712,-0.21359862,-4.856464])
# embeddings.append([0.750622,-0.7132483,-1.2707405,-0.19877918,14.4466,1.3527036,-2.9070518,3.2331638,-3.613355,0.078160346,-5.9566245,-22.410223,-27.49646,13.948006,0.009050498,6.310787])
# embeddings.append([2.0344,-0.18211967,-1.016011,-0.58528906,9.130649,0.6399375,3.0131373,6.2298174,-0.51527363,-0.891654,-5.2474966,-22.75474,-25.72084,13.634716,0.80226827,11.896404])
# embeddings.append([-0.9266835,-0.6330202,-1.6333201,0.63965774,6.352248,0.66435474,0.8719162,2.4231133,1.841162,1.8156424,-7.528089,-24.090658,-28.71157,13.208413,-2.377902,13.784845])
# embeddings.append([-1.807375,-0.2934243,-0.53428394,1.122416,3.113141,0.43747795,2.2722063,2.1431649,-1.4136422,-0.55535513,-7.9582887,-23.233164,-28.072113,14.592276,-2.953101,14.686382])
# embeddings.append([-0.095981285,0.007204056,-0.6657414,-1.8221579,1.1506329,-0.12960353,1.6366785,0.75239086,-2.0837288,0.45380038,-6.7266593,-24.581785,-28.539667,14.750387,-2.41665,16.084827])
# embeddings.append([0.8919491,0.50149125,-0.11360723,0.42947873,-0.5063789,0.54685724,0.14096078,0.09657806,-0.5318501,0.5202051,-6.641752,-25.089462,-28.634327,13.858293,0.71471775,18.007915])
# embeddings.append([-0.93191594,-0.41936427,-0.07612151,-0.059824474,-1.2776589,1.6989226,1.0947989,-2.3466268,-1.4157791,2.0586221,-7.58253,-28.525173,-30.30743,14.7991085,2.3285377,18.076374])
# embedding = [175.43193,66.27316,-320.3051,158.65048,-6.7693815,79.36069,123.895325,215.09938,313.30115,166.8979,78.46089,-311.2562,-310.2758,-318.7852,67.54407,5.2377653]

for i in range(10):
    plt.bar(range(3), embeddings[i])
    plt.title("forward_"+str(i+1))
    plt.show()
    # plt.savefig("forward_"+str(i+1)+".png")