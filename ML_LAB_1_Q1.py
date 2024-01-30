def countpairs(arr, n, k):
    #initializing the count
    count=0
    
    #checking their sums for all the possible pairs
    for i in range(0,n):
        for j in range(i+1,n):
            if arr[i]+arr[j] == k:
                count+=1

    return count

arr=[2,7,4,1,3,6]
n=len(arr)
k=10
#result
print("count of pairs is ",countpairs(arr,n,k))