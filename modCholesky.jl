eta = 0.0000001 #betweeen 0 to 1


function modchol(M, eta=0.0000001)
  #Check for square matrix
  m,n = size(M)
  if m!=n
    throw(DimensionMismatch("M should be a square Matrix"))
  end

  #Making Matrix sparse matrix
  #M = sparse(M)

  #initialization
  M_prev = M
  L = sparse(zeros((n,n)))
  beta = maximum(diag(M))   #beta = max_i(1,...,m)M_ii

  for i = 1:m
    if diag(M_prev)[i] <= beta*eta
      print("Skipping")
      #skip this elimination step
      E_curr = get_E_for_modchol(M_prev, i)
      M_curr = M_prev - E_curr
      #@show(i, E_curr)
    else
      #perform usual Cholesky elimination step
      L[i,i] = sqrt(M_prev[i,i])
      M_curr = sparse(zeros(m,n))

      for j = i+1:m
        L[j,i] = M_prev[i,j] / L[i,i]
      end

      for j = i+1:m
        for k = i+1:m
          M_curr[j,k] = M_prev[j,k] - L[j,i]*L[k,i]
        end
      end
    end
    #@show(i,M_prev, M_curr, L)
    M_prev = M_curr
  end
  @show(L)
  return L
end

function get_E_for_modchol(M, index)
  m,n = size(M)
  E = sparse(zeros((m,n)))
  for i = index:m
    E[index,i] = M[index,i]
    E[i,index] = M[i,index]
  end
  return E
end

#M = [25 15 -5;
#      15 18 0;
#      -5 0 11]

#modchol(M)
#get_E_for_modchol(M, index)

