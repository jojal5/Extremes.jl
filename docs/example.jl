using Extremes, DataFrames, Distributions, Gadfly, Dates. Mamba

data = load("wooster")

x = collect(Date(1983,1,1):Day(1):Date(1987,12,31))

data[!,:Date] = x
select!(data, [:Date, :Temperature])

data

plot(data, x=:Date, y=:Temperature, Geom.point)

df = copy(data)

df[!,:Temperature] = -data[:,:Temperature]


filter!(row -> month(row.Date) ∈ (1,2,11,12), df)

plot(df, x=:Date, y=:Temperature, Geom.point)

y = df[:,:Temperature]

threshold =  -10

cluster = getcluster(y, threshold, runlength=4)

z = maximum.(cluster) .- threshold

fm = gpfit(z)

r = returnlevel(fm, threshold, size(df,1), 120, 100, .95)

r.value[]
r.cint[]






data = load("dowjones")

X = data[:,:Index]

X̃ = 100*(log.(X[2:end]) - log.(X[1:end-1]))

df = DataFrame(Date = data[2:end, :Date], Variation = X̃)

plot(df, x=:Date, y=:Variation, Geom.line)

threshold = 2.0

cluster = getcluster(df[:,:Variation], threshold, runlength=4)

z = maximum.(cluster) .- threshold

fm = gpfit(z)


data = load("fremantle")

fm = gevfitbayes(data, :SeaLevel, locationcovid = [:Year, :SOI])

p = plot(fm.sim[:,1,1])

p[2]


p = plot(fm.sim)

hstack([p[1], p[2]])





# Lis all files in a directory
filenames = String[]


for (root, dirs, files) in walkdir("src/validationplots/")
    for file in files
        push!(filenames, joinpath(root, file))
        # println(joinpath(root, file)) # path to files
    end
end
