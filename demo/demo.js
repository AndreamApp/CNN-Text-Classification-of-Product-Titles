
let ws = null;
let demo = null;

const pendingFilter = item => item.status == 'pending';
const waitingFilter = item => item.status == 'waiting';
// 不查询空项 和 正在查询的项
const queryFilter = item => ['waiting', 'success', 'failed'].includes(item.status);

function setState(state) {
    if('已连接' == state) {
        $('#state').removeClass('err').removeClass('processing').addClass('connected');
        $('#state_text').text('已连接');
    }
    else if('未连接' == state) {
        $('#state').removeClass('processing').removeClass('connected').addClass('err');
        $('#state_text').text('未连接');
    }
    else if('正在预测' == state) {
        $('#state').removeClass('err').removeClass('connected').addClass('processing');
        $('#state_text').text('正在预测');
    }
}

function setMessage(msg) {}

function hashCode(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        let c = str.charCodeAt(i);
        hash = ((hash<<5) - hash) + c;
        hash = hash & hash; // Convert to 32bit integer
    }
    return hash;
}

function uniqId() {
    return (new Date()).getTime();
}

function download(data, filename, type) {
    var file = new Blob([data], {type: type});
    if (window.navigator.msSaveOrOpenBlob) // IE10+
        window.navigator.msSaveOrOpenBlob(file, filename);
    else { // Others
        var a = document.createElement("a"),
                url = URL.createObjectURL(file);
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(function() {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);  
        }, 0); 
    }
}

function initSocket(postData, postState) {
    ws = new WebSocket("ws://127.0.0.1:8765/");
    ws.onmessage = function (event) {
        // console.log(event.data);
        demo.decodeResults(event.data);
        setState('已连接');
    };
    ws.onopen = (event) => {
        setState('已连接');
        if(postData) {
            sendSocket(postData, postState);
        }
    };
    ws.onclose = (event) => {
        setState('未连接');
        ws = null;
    };
    ws.onerror = (event) => {
        setState('未连接');
        alert('连接模型失败！\n请先启动后端模型，执行： `python3 server.py`');
        ws = null;
    };
}

function sendSocket(data, state) {
    // TODO: streaming for big file
    console.log('<', data)
    if(ws != null && ws.readyState == 1) {
        if(state) {
            setState(state);
        }
        ws.send(data);
    }
    else {
        initSocket(data, state);
    }
}

demo = new Vue({
    el: '#content',
    data: {
        items: [
            {
                id: uniqId(),
                type: "text", // text or file
                title: "专用奥迪A3A4LA5A6L前杠防撞胶条 侧裙胶条前唇装饰车身碳纤胶条", // title or filename
                status: "waiting", // empty, waiting, pending, success, failed
                result: "等待分类", // classify result or output file
                file_content: "",
                time: "",
            }
        ],
        colors: [
            '#4798E5', '#81A3E2', '#454851', '#A99FAD', '#7C7342', '#93748A', '#969176'
        ],
        modelSelected: 'TextCNN-CHAR'
    },
    methods: {
        encodeItemsQuery: function() {
            let fieldFilter = ['cmd', 'model', 'items', 'id', 'type', 'title', 'file_content'];
            let obj = {
                'cmd': 'query',
                'model': this.modelSelected,
                'items': this.items.filter(queryFilter)
            };
            console.log(obj);
            let data = JSON.stringify(obj, fieldFilter);
            return data;
        },
        encodeItemsClear: function(clearItems) {
            // only clear pending items
            let fieldFilter = ['cmd', 'items', 'id'];
            let obj = {
                'cmd': 'clear',
                'items': clearItems.filter(pendingFilter)
            };
            let data = JSON.stringify(obj, fieldFilter);
            return data;
        },
        encodeItemsClearAll: function() {
            let obj = {
                'cmd': 'clearAll'
            };
            let data = JSON.stringify(obj);
            return data;
        },

        decodeResults: function(results) {
            console.log('>', results);
            let data = JSON.parse(results);
            if('result' == data['cmd']) {
                data['results'].forEach(res => {
                    demo.items.forEach((item, index) => {
                        if(res.id == item.id) {
                            item.status = res.status;
                            item.result = res.result;
                            item.time = res.time;
                        }
                    })
                })
            }
        },
        // 添加一行文本
        addLine: function() {
            this.items.push(
            {
                id: uniqId(),
                type: "text", // text or file
                title: "", // title or filename
                status: "empty",
                result: "等待分类", // classify result or output file
                file_content: "",
                time: ""
            });
            setTimeout(function() {
                let titles = document.querySelectorAll('.product-title');
                window.scrollTo(0,document.body.scrollHeight + 200);
                titles[titles.length-1].focus();
            });
        },
        // 添加小文件，保存文件内容
        addFile: function(filename, content) {
            this.items.push(
            {
                id: uniqId(),
                type: "file", // text or file
                title: filename, // title or filename
                status: "waiting",
                result: "等待分类", // classify result or output file
                file_content: btoa(encodeURIComponent(content.replace(/\r/g, ''))), // base64编码
                time: ""
            });
            setTimeout(function() {
                let titles = document.querySelectorAll('.product-title');
                window.scrollTo(0,document.body.scrollHeight + 200);
                titles[titles.length-1].focus();
            });
        },
        // 添加大文件，只保存文件名
        addPath: function(filename) {
            this.items.push(
            {
                id: uniqId(),
                type: "path",
                title: filename,
                status: "waiting",
                result: "等待分类",
                file_content: "",
                time: ""
            });
            setTimeout(function() {
                let titles = document.querySelectorAll('.product-title');
                window.scrollTo(0,document.body.scrollHeight + 200);
                titles[titles.length-1].focus();
            });
        },
        clearLines: function() {
            // cancle all tasks on cloud
            if(this.items.filter(pendingFilter).length > 0) {
                let request = this.encodeItemsClearAll();
                sendSocket(request);
            }
            this.items = [];
            this.addLine();
        },
        removeLine: function(item) {
            let index = this.items.indexOf(item);
            if(index > -1) {
                // cancle task on cloud
                if(item.status == 'pending') {
                    let request = this.encodeItemsClear([item]);
                    sendSocket(request);
                }
                this.items.splice(index, 1);
            }
        },
        onLineChanged: function(item) {
            // let index = this.items.indexOf(item);
            // if(index > -1) {
            //     if(item.title.trim()) {
            //         this.items[index].status = 'waiting';
            //     }
            //     else {
            //         this.items[index].status = 'empty';
            //     }
            // }
        },
        selectFile: function() {
            document.getElementById('fileSelector').click();
        },
        uploadFile: function(event) {
            let files = event.target.files;
            console.log(files);
            if(files.length > 0) {
                let file = files[0]; // name, size
                console.log(file);
                if(file.size < 1024 * 100) {
                    let fr = new FileReader();
                    fr.onload = function() {
                        // console.log(fr.result);
                        demo.addFile(file.name, fr.result);
                        // 如果不清空值，下次选择同一文件时不会触发onchange
                        document.getElementById('fileSelector').value = '';
                    };
                    fr.readAsText(file, 'gbk');
                }
                else {
                    alert('文件大小超过100K，请确保该文件和server.py位于同一路径！');
                    demo.addPath(file.name);
                    // 如果不清空值，下次选择同一文件时不会触发onchange
                    document.getElementById('fileSelector').value = '';
                }
            }
        },
        predict: function() {
            // 重置每一项的状态
            this.items.forEach(item => {
                if(item.title) {
                    if(item.status == 'empty') {
                        item.status = 'waiting';
                    }
                }
                else {
                    item.status = 'empty';
                }
            });
            if(this.items.filter(queryFilter).length == 0) {
                alert('请添加新的数据后再预测！');
            }
            else {
                let data = this.encodeItemsQuery();
                sendSocket(data, '正在预测');
            }
        },
        exportFile: function() {
            let data = '';
            let cnt = 0;
            this.items.forEach(function(item) {
                if(item.type == 'text' && item.result != '等待分类' && item.title) {
                    data += item.title + '\t' + item.result + '\n';
                    cnt++;
                }
            });
            if(cnt <= 0) {
                alert('没有可导出的数据！请先输入数据并点击预测后再导出');
            }
            else {
                download(data, 'export.txt', 'text/plain');
            }
        },
        resultClass: function(result) {
            let style = {
                background: '#81BF76',
                color: '#f8f8f8'
            };
            if(result == '等待分类') {
                style.background = '#81BF76';
            }
            else {
                let key = result.split('-')[0];
                let index = Math.abs(hashCode(key)) % this.colors.length;
                // console.log(hashCode(key), index);
                style.background = this.colors[index];
            }
            return style;
        }
    }
});

setTimeout(() => {
    sendSocket('{"cmd":"modelList"}');
}, 1000);
