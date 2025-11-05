classdef ParamHashSet < handle
    % ParamHashSet - Efficient storage and lookup for fixed-dimension parameter combinations
    %
    % This class uses containers.Map with string-encoded keys for fast add/check operations.
    %
    % Example:
    %   s = ParamHashSet();
    %   s.add([1 1 1]);
    %   tf = s.contains([1 1 1]); % returns true
    %   tf = s.contains([1 1 2]); % returns false
    %
    % -------------------------------------------------------------------------
    % Author : Xi Guo
    % Email  : xiguo@my.swjtu.edu.cn
    % Date   : 2025-10-27
    % -------------------------------------------------------------------------

    properties (Access = private)
        map  % containers.Map to store parameter combinations
    end

    methods
        function obj = ParamHashSet()
            % Constructor: initialize empty hash set
            obj.map = containers.Map('KeyType','char','ValueType','logical');
        end

        function add(obj, param)
            % add - Add a parameter combination to the set
            key = obj.encodeParam(param);
            obj.map(key) = true;
        end

        function tf = contains(obj, param)
            % contains - Check if a parameter combination exists
            key = obj.encodeParam(param);
            tf = isKey(obj.map, key);
        end

        function n = size(obj)
            % size - Return the number of stored combinations
            n = obj.map.Count;
        end
    end

    methods (Access = private)
        function key = encodeParam(~, param)
            % encodeParam - Convert a vector to a unique string key
            key = mat2str(param);
            % key = num2str(param);
        end
    end
end
