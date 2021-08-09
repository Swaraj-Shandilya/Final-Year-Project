const express    = require('express'),
      app        = express(),
      bodyParser = require('body-parser'),
      fs         = require('fs'),
      parse      = require('csv-parse/lib/sync'),
      {spawn}    = require('child_process');

app.set("view engine", "ejs");
app.use(express.static(__dirname + "/public"));
app.use(bodyParser.urlencoded({extended: true}));
app.use(bodyParser.json());

app.get("/",function(req,res){
    res.render("index",{url:'',flag:false})
});

app.post("/visualize",function(req,res) {
    console.log(req.body);
    fs.writeFileSync('analyzer/sample-links.txt',req.body.url);
    const reviews = spawn('python3', ['analyzer/reviews.py']);
    reviews.on('close', (code) => 
    {
        console.log("Scrapping Completed");
        const analyzer = spawn('python3',['analyzer/evaluator.py'])
        analyzer.on('close',(ecode) => {
            console.log("Files Generated");
            const input = fs.readFileSync('analyzer/out.csv');
            const records = parse(input, {
            columns: true,
            skip_empty_lines: true,
            });
            res.render("index",{url:req.body.url,flag:true,records});
        });
    });
});

app.listen(3000,function() {
    console.log('Backend for Sentiment Analysis is active at http://localhost:3000');
});