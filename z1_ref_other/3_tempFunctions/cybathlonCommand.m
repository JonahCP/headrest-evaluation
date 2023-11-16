% Exexute following lines to initialize
% udpSocket = udp('127.0.0.1', 5555);
% fopen(udpSocket);

% Execute 
% following lines to close 
% fclose(udpSocket);

function cybathlonCommand(udpSocket, type)
    switch(type)
        case 'left'
            command = 11;
        case 'light'
            command = 12;
        case 'right'
            command = 13;            
    end
    fwrite(udpSocket, command);
end