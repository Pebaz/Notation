from nndt import Array, String


Array[3, String[3]](['ABC', '10', '!']).to()
Array[3, String[3]](['ABC', '10', '!']).as_layer()
Array[3, String[3]].from_layer(Array[3, String[3]](['ABC', '10', '!']).as_layer())
Array[3, String[3]].from_layer(Array[3, String[3]](['ABC', '10', '!']).as_layer()).to()
